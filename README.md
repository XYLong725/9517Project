# 🛰️ Aerial Landscape Classification (COMP9517 Project)

This repository presents our project for **COMP9517**, focusing on **aerial landscape image classification** using both traditional and deep learning methods. Three approaches are implemented and compared in terms of accuracy, robustness, and visualization interpretability.

---

## 📁 Dataset

We use the **SkyView** dataset, which consists of **15 aerial scene categories**, each with **800 images (256×256 resolution)**. The dataset is compiled from [AID](https://captain-whu.github.io/AID/) and [NWPU-RESISC45](https://www.tensorflow.org/datasets/catalog/resisc45).

---

## 🛠️ Requirements

- Python 3.9+
- PyTorch ≥ 2.0
- torchvision
- numpy
- scikit-learn
- matplotlib
- seaborn (for visualization)

> All experiments are conducted using Google Colab or AutoDL with CUDA support.

---

## 🧠 Models Implemented

| Approach            | Features             | Classifier     | Accuracy | Macro F1 |
|---------------------|----------------------|----------------|----------|----------|
| SIFT + BoW + SVM    | SIFT keypoints       | Linear SVM     | ~85%     | ~0.83    |
| ResNet18 (baseline) | Raw Pixels 224×224   | Pretrained CNN | ~95.5%   | ~0.95    |
| EfficientNet B0     | Raw Pixels 224×224   | Pretrained CNN | ~97.5%   | ~0.97    |

### 🔁 Augmentation Ablation (EfficientNet B0)
We tested three data augmentation strategies:

- `randomcrop`
- `colorjitter`
- `augmix` ✅ (best robustness under distortion)

---


