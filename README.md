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
├── model_output/                  # Trained model .pth files (not uploaded due to size)
├── notebooks/                     # Jupyter notebooks for training, evaluation, and visualization
├── results/                       # Saved images: confusion matrices, loss curves, robustness plots
├── score_*.txt                    # Accuracy and F1 logs
├── acc_*.npy / loss_*.npy         # Per-epoch metrics
└── README.md                      # Project documentation (this file)
```

---

## Implemented Methods
We compared traditional and deep learning methods:

| Method        | Feature Type        | Model         | Accuracy | Macro F1 |
|---------------|---------------------|----------------|----------|----------|
| BoW + SVM     | SIFT                | SVM (linear)   | ~85%     | ~0.83    |
| EfficientNet  | Raw Pixels (224x224)| Pretrained B0  | ~97.5%   | ~0.97    |
| ResNet18      | Raw Pixels          | Pretrained     | ~95.5%   | ~0.95    |

> We also tested EfficientNet with three augmentations: `randomcrop`, `colorjitter`, `augmix`, and found **AugMix** gives better robustness under noise.

---

## Training & Evaluation

### Train Your Own Model
```bash
python train_efficientnet_augmentation_ablation.ipynb
python train_resnet.ipynb
python train_sift_svm_bow.ipynb
```

### Evaluate Robustness
```bash
python robustness_test.ipynb
```
> Includes Clean / Noise / Occlusion test and plot generation.

### Visualize Attention (GradCAM)
```bash
python gradcam_visualization.ipynb
```
> Visualizes where the model focuses using GradCAM.

---

## Visualization Results
| Type                 | File                          |
|----------------------|-------------------------------|
| Accuracy vs. Epoch   | training_accuracy_comparison.png |
| Loss vs. Epoch       | training_loss_comparison.png  |
| Confusion Matrix     | confusion_matrix_resnet.png / confusion_matrix_efficientnet.png |
| Robustness Test      | robustness_*.png              |
| Final Comparison     | robustness_comparison_all_models.png |

---

## Notes
- All `.pth` files are excluded due to GitHub's 25MB limit. Please contact the authors to obtain.
- All evaluation metrics and plots can be reproduced using the `.npy` logs included.

---

## Credits
This project is completed by **Xiangyun Long** for COMP9517 2025. Inspired by [Fashion-MNIST benchmark repo](https://github.com/xuehaouwa/Fashion-MNIST).

---

## License
MIT License

