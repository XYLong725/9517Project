# 9517Project: Aerial Landscape Classification

This repository contains our project for COMP9517, focusing on aerial landscape image classification using traditional and deep learning methods. We implemented three different approaches and conducted extensive experiments including augmentation, robustness testing, and visualization.

## Dataset
The dataset used is the **SkyView** dataset, containing 15 categories of aerial landscape images with 800 images per class.

---

## Requirements
- Python 3.9+
- PyTorch >= 2.0
- torchvision
- numpy
- matplotlib
- scikit-learn
- seaborn (for visualization)

---

## Project Structure
```
9517Project/
├── model_output/                  # Trained model .pth files (hosted under GitHub Releases)
├── notebooks/                     # Jupyter notebooks for training, evaluation, and visualization
├── score/                         # Final accuracy and macro-F1 text logs
├── confusion_matrix/             # Visualized confusion matrices (png)
├── training_loss_comparison/     # Loss over epochs (png + npy)
├── training_accuracy_comparison/ # Accuracy over epochs (png + npy)
├── robustness_results/           # Noise/occlusion robustness comparison (png)
├── pca_bow/                       # PCA plots for SIFT-BoW histograms
├── gradcam_outputs/              # GradCAM visualizations
└── README.md                      # Project documentation (this file)
```

---

## Implemented Methods
We compared traditional and deep learning methods:

| Method            | Feature Type        | Model         | Accuracy | Macro F1 |
|-------------------|---------------------|----------------|----------|----------|
| BoW + SVM         | SIFT                | SVM (linear)   | ~0.51    | ~0.48    |
| EfficientNet B0   | Raw Pixels (224x224)| Pretrained     | ~0.98    | ~0.98    |
| ResNet18          | Raw Pixels          | Pretrained     | ~0.90    | ~0.89    |

### Augmentation Methods (EfficientNet B0):
| Strategy     | Accuracy | Macro F1 |
|--------------|----------|----------|
| Baseline     | 0.98     | 0.98     |
| RandomCrop   | 0.96     | 0.96     |
| ColorJitter  | 0.96     | 0.96     |
| AugMix       | 0.95     | 0.95     |

> EfficientNet Baseline performs best on clean data, while **AugMix** gives better robustness under noise/occlusion.

---

## Training & Evaluation

### Train Models
```bash
# EfficientNet + Augmentation
python train_efficientnet_augmentation_ablation.ipynb

# ResNet18
python train_resnet.ipynb

# SIFT + BoW + SVM
python train_sift_svm_bow.ipynb
```

### Evaluate Robustness
```bash
python robustness_test.ipynb
```
> Includes Clean / Noise / Occlusion tests + comparison plot `robustness_comparison_all_models.png`

### Visualize Model Attention (GradCAM)
```bash
python gradcam_visualization.ipynb
```
> See where the model focuses when making predictions (GradCAM heatmaps).

### Confusion Matrices
```bash
python confusion_matrix_resnet.ipynb
python confusion_matrix_efficientnet.ipynb
```
> Compare ResNet and EfficientNet confusion trends.

---

## Visualization Results
| Type                   | File                                     |
|------------------------|------------------------------------------|
| Accuracy vs. Epoch     | training_accuracy_comparison.png         |
| Loss vs. Epoch         | training_loss_comparison.png             |
| Confusion Matrix       | confusion_matrix_resnet.png / efficientnet.png |
| Robustness Test        | robustness_randomcrop.png, ...           |
| Final Comparison       | robustness_comparison_all_models.png     |
| SIFT BoW PCA           | pca_bow_*.png                            |
| GradCAM Heatmaps       | gradcam_outputs/*.png                    |

---

## Model Checkpoints
All model `.pth` files are hosted via **GitHub Releases** due to size restrictions. Please check the [Releases page](https://github.com/XYLong725/9517Project/releases) for downloads.

---


