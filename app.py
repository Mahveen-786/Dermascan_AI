import gradio as gr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from huggingface_hub import hf_hub_download
import json

REPO_ID     = "mahveen123/dermascan-ai"
NUM_CLASSES = 7
CLASS_NAMES = ['actinic_keratosis','basal_cell_carcinoma','benign_keratosis',
               'dermatofibroma','melanoma','melanocytic_nevus','vascular_lesion']
MELANOMA_IDX = 4

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

transform_32 = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
transform_224 = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class CNNFromScratch(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # 0
            nn.BatchNorm2d(32),                # 1
            nn.ReLU(),                         # 2
            nn.Conv2d(32, 32, 3, padding=1),  # 3
            nn.BatchNorm2d(32),                # 4
            nn.ReLU(),                         # 5
            nn.MaxPool2d(2),                   # 6
            nn.Conv2d(32, 64, 3, padding=1),  # 7
            nn.BatchNorm2d(64),                # 8
            nn.ReLU(),                         # 9
            nn.Conv2d(64, 64, 3, padding=1),  # 10
            nn.BatchNorm2d(64),                # 11
            nn.ReLU(),                         # 12
            nn.MaxPool2d(2),                   # 13
            nn.Conv2d(64, 128, 3, padding=1), # 14
            nn.BatchNorm2d(128),               # 15
            nn.ReLU(),                         # 16
            nn.Conv2d(128, 128, 3, padding=1),# 17
            nn.BatchNorm2d(128),               # 18
            nn.ReLU(),                         # 19
            nn.MaxPool2d(2),                   # 20
            nn.Conv2d(128, 256, 3, padding=1),# 21
            nn.BatchNorm2d(256),               # 22
            nn.AdaptiveAvgPool2d(1),           # 23
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 7),
        )
    def forward(self, x):
        return self.classifier(self.features(x))

def create_resnet18():
    m = models.resnet18(weights=None)
    m.conv1   = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    m.maxpool = nn.Identity()
    m.fc      = nn.Linear(512, NUM_CLASSES)
    return m

class ViTFromScratch(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=192,
                 num_heads=3, num_layers=6, num_classes=7, dropout=0.1):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_embed  = nn.Conv2d(3, embed_dim, patch_size, stride=patch_size)
        self.cls_token    = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed    = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim) * 0.02)
        self.pos_drop     = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation='gelu', batch_first=True,
            norm_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )
    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1) + self.pos_embed
        x = self.pos_drop(x)
        x = self.norm(self.transformer(x))
        return self.head(x[:, 0])

def create_vit_b16():
    import timm
    return timm.create_model('vit_base_patch16_224', pretrained=False, num_classes=NUM_CLASSES)

MODEL_CFG = {
    "CNN Scratch":          ("dermascan_cnn_scratch.pth",  CNNFromScratch,  transform_32),
    "ResNet-18 Pretrained": ("dermascan_resnet18_pt.pth",  create_resnet18, transform_32),
    "ViT Scratch":          ("dermascan_vit_scratch.pth",  ViTFromScratch,  transform_32),
    "ViT-B16 Pretrained":   ("dermascan_vit_b16_pt.pth",   create_vit_b16,  transform_224),
}

def load_model(filename, create_fn):
    path = hf_hub_download(repo_id=REPO_ID, filename=filename)
    ck   = torch.load(path, map_location="cpu", weights_only=False)
    m    = create_fn()
    m.load_state_dict(ck["model_state_dict"])
    m.eval()
    return m

print("Loading models...")
LOADED = {}
for name, (fname, fn, _) in MODEL_CFG.items():
    try:
        LOADED[name] = load_model(fname, fn)
        print(f"✅ {name}")
    except Exception as e:
        print(f"❌ {name}: {e}")
        LOADED[name] = None

CLASS_INFO = {
    "actinic_keratosis":    "Pre-cancerous UV lesion. Monitor and consult a dermatologist.",
    "basal_cell_carcinoma": "Common skin cancer. Treatable if caught early.",
    "benign_keratosis":     "Non-cancerous growth. No treatment usually needed.",
    "dermatofibroma":       "Benign fibrous nodule. Harmless in most cases.",
    "melanoma":             "Malignant skin cancer. Consult a dermatologist immediately.",
    "melanocytic_nevus":    "Common mole. Monitor for changes (ABCDE rule).",
    "vascular_lesion":      "Blood vessel abnormality. Usually benign.",
}

def predict(image, model_name):
    if image is None:
        return "", "", gr.update(value=None, visible=False)
    model = LOADED.get(model_name)
    if model is None:
        return "<p style='color:red'>Model failed to load.</p>", "", gr.update(visible=False)
    _, _, tfm = MODEL_CFG[model_name]
    img = tfm(image).unsqueeze(0)
    with torch.no_grad():
        probs = F.softmax(model(img), dim=1)[0]
    pred_idx   = probs.argmax().item()
    pred_name  = CLASS_NAMES[pred_idx]
    mel_prob   = probs[MELANOMA_IDX].item()
    confidence = float(probs[pred_idx]) * 100
    if mel_prob > 0.4:
        risk_html = f'<div style="background:#2d0f0f;border:2px solid #dc2626;border-radius:10px;padding:16px 20px;color:#fca5a5;font-weight:600;">🔴 HIGH RISK — Melanoma probability: {mel_prob*100:.1f}%<br><span style="font-weight:400;font-size:0.85rem;">Consult a dermatologist immediately.</span></div>'
    elif mel_prob > 0.2:
        risk_html = f'<div style="background:#2d2200;border:2px solid #d97706;border-radius:10px;padding:16px 20px;color:#fcd34d;font-weight:600;">🟡 MEDIUM RISK — Melanoma probability: {mel_prob*100:.1f}%<br><span style="font-weight:400;font-size:0.85rem;">Monitor closely and consider professional evaluation.</span></div>'
    else:
        risk_html = f'<div style="background:#0f2d1a;border:2px solid #16a34a;border-radius:10px;padding:16px 20px;color:#86efac;font-weight:600;">🟢 LOW RISK — Melanoma probability: {mel_prob*100:.1f}%<br><span style="font-weight:400;font-size:0.85rem;">Appears benign. Continue regular skin checks.</span></div>'
    desc = CLASS_INFO.get(pred_name, "See a dermatologist for advice.")
    pred_html = f'<div style="background:#1a1f2e;border:1px solid #2a3a5c;border-radius:12px;padding:20px;margin-bottom:12px;"><div style="font-size:1.3rem;font-weight:700;color:#e2e8f0;margin-bottom:6px;">{pred_name.replace("_"," ")}</div><div style="color:#94a3b8;font-size:0.9rem;margin-bottom:8px;">Confidence: <strong>{confidence:.1f}%</strong></div><div style="color:#64748b;font-size:0.85rem;border-top:1px solid #2a3a5c;padding-top:10px;">{desc}</div></div>'
    scores = {CLASS_NAMES[i].replace("_", " "): float(probs[i]) for i in range(NUM_CLASSES)}
    return pred_html, risk_html, gr.update(value=scores, visible=True)

with gr.Blocks(title="DermaScan AI") as demo:
    gr.HTML("""<div style="display:flex;align-items:center;gap:16px;background:#1a1f2e;border:1px solid #2a3a5c;border-radius:16px;padding:24px 32px;margin-bottom:16px;"><div style="font-size:48px;">🔬</div><div><h1 style="margin:0;font-size:2rem;font-weight:700;background:linear-gradient(90deg,#4f8ef7,#a78bfa);-webkit-background-clip:text;-webkit-text-fill-color:transparent;">DermaScan AI</h1><p style="color:#8892a4;margin:4px 0 0;">Melanoma &amp; Skin Lesion Classifier · CNN · ResNet-18 · ViT-Scratch · ViT-B16</p></div></div>""")
    gr.HTML("""<div style="background:#1e1a0e;border-left:4px solid #f59e0b;border-radius:8px;padding:12px 16px;color:#fbbf24;font-size:0.88rem;margin-bottom:16px;">⚠️ <strong>Medical Disclaimer:</strong> For educational and research purposes only. Not a substitute for professional medical advice.</div>""")
    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML('<p style="font-weight:600;color:#a78bfa;">📤 Upload dermoscopy image</p>')
            img_input = gr.Image(type="pil", label="Skin lesion image", height=300)
            model_selector = gr.Dropdown(choices=list(MODEL_CFG.keys()), value="ResNet-18 Pretrained", label="🤖 Select Model")
            submit_btn = gr.Button("🔍 Analyse Lesion", variant="primary", size="lg")
            gr.HTML("""<div style="background:#111827;border:1px solid #2a3a5c;border-radius:10px;padding:14px 16px;font-size:0.85rem;color:#9ca3af;margin-top:12px;"><strong>💡 Tips:</strong><ul style="margin:6px 0 0 16px;"><li>Use clear dermoscopy or close-up photos</li><li>Centre the lesion in frame</li><li>Ensure good lighting</li></ul></div>""")
        with gr.Column(scale=1):
            gr.HTML('<p style="font-weight:600;color:#a78bfa;">📊 Results</p>')
            out_pred  = gr.HTML()
            out_risk  = gr.HTML()
            out_chart = gr.Label(label="Class Probabilities", num_top_classes=7, visible=False)
    submit_btn.click(fn=predict, inputs=[img_input, model_selector],
                     outputs=[out_pred, out_risk, out_chart])
    gr.HTML("""<div style="text-align:center;color:#4b5563;font-size:0.82rem;padding:20px;border-top:1px solid #1f2937;margin-top:16px;">CNN · ResNet-18 · ViT-Scratch · ViT-B16 · PyTorch + Gradio + timm</div>""")

demo.launch()
