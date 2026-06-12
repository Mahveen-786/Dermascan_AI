# DermaScan AI 🔬

Melanoma and skin lesion classifier comparing 4 model architectures on HAM10000 (DermaMNIST).
## Demo
[![DermaScan Demo](https://img.youtube.com/vi/Gg78xnABCMY/0.jpg)](https://youtube.com/watch?v=Gg78xnABCMY)

> Click to watch the 2-minute demo

## Live Demo
 [Try it on HuggingFace Spaces](https://huggingface.co/spaces/mahveen123/dermascan)

## Research Question
> *Does pretraining matter more than architecture for small medical datasets?*

## Models Compared
| Model | Pretraining | Input Size |
|---|---|---|
| CNN from scratch | ❌ None | 32×32 |
| ResNet-18 | ✅ ImageNet | 32×32 |
| ViT from scratch | ❌ None | 32×32 |
| ViT-B/16 | ✅ ImageNet-21k | 224×224 |

## Key Finding
Pretrained models significantly outperform scratch-trained models on small medical datasets.
ViT from scratch fails without sufficient data — ViT-B/16 pretrained achieves the highest melanoma sensitivity.

## Tech Stack
PyTorch · timm · Gradio · HuggingFace Spaces 

---
*For educational and research purposes only. Not a substitute for medical advice.*



