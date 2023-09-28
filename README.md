# Image and Video Classification — Deep Learning with CNNs, MobileNetV2 & ResNet-50

> MSc Data Science & Analytics dissertation project (Royal Holloway, University of London, 2023) exploring image segmentation and classification across multiple neural network architectures — from custom CNNs to an improved ResNet-50 — with systematic hyperparameter tuning.

---

## Project Background

This project investigates how model architecture and hyperparameter choices affect performance across image segmentation and classification tasks of varying complexity. It compares lightweight custom CNNs against transfer-learning approaches (MobileNetV2, ResNet-50) on datasets ranging from simple synthetic shapes to real-world medical imagery (malaria-infected blood smears) and the Oxford-IIIT Pet dataset.

**Key result:** an improved ResNet-50 architecture achieved **97.42% validation accuracy** on the malaria classification task after hyperparameter tuning, while a much simpler custom CNN reached **99.45% training accuracy / 99.23% validation accuracy** on MNIST — demonstrating that architectural complexity should match dataset complexity rather than always favoring the largest available model.

---

## Repository Structure

```
image-video-classification/
│
├── Image_and_Video_Classification_Report.pdf   # Full dissertation report (44 pages)
├── Segmentation.ipynb                           # Image segmentation experiments
├── Classification_part_1.ipynb                  # IRIS, MNIST, MobileNetV2, custom CNNs
├── Classification_part_2.ipynb                  # Malaria dataset, hyperparameter tuning, ResNet-50
└── README.md
```

---

## Datasets Used

| Dataset | Task | Size |
|---|---|---|
| Synthetic circles (generated) | Image segmentation | 1,000 images, 128×128 |
| Oxford-IIIT Pet | Image segmentation (real-world) | ~7,000 images |
| IRIS | Classification baseline | 150 samples, 4 features |
| MNIST | Digit classification | 60,000 train / 10,000 test |
| Malaria (TensorFlow Datasets) | Binary medical image classification | ~27,000 blood smear images, 70×70 |

---

## Experiments & Results

### 1. Image Segmentation

| Experiment | Architecture | Result |
|---|---|---|
| Custom ConvNet | Conv2D + MaxPooling + Conv2DTranspose | Training loss 0.4606 → 0.0024 over 10 epochs; very low validation loss (0.0027), but synthetic data may be too simple to validate robustness |
| MobileNetV2-based | Pre-trained MobileNetV2 + custom decoder | Training loss plateaued at ~0.26; struggled to adapt to synthetic data |
| U-Net (MobileNetV2 backbone) | Pre-trained encoder + pix2pix-style decoder | Real-world Oxford-IIIT Pet dataset; ~90% validation accuracy after 20 epochs, with signs of mild overfitting |

### 2. Image Classification

| Experiment | Architecture | Result |
|---|---|---|
| IRIS baseline | 3-layer dense network | 96.67% test accuracy |
| MobileNetV2 on MNIST | Transfer learning | Only ~41% validation accuracy — oversized architecture, poor fit, high compute cost (~1555s/epoch) |
| Custom CNN (2 conv blocks) | Simple CNN + dropout | **99.45% train / 99.23% validation accuracy**, ~135s/epoch — best efficiency-to-accuracy ratio |
| Complex CNN (3 conv blocks, BatchNorm, L1/L2) | Deeper regularized CNN | ~98.85% validation accuracy but higher compute cost and signs of overfitting by epoch 5 |

### 3. Malaria Detection (Hyperparameter Tuning)

A grid search was run across optimizer (Adam/SGD), dropout rate (0.2/0.3/0.4), dense units (128/256/512), learning rate (0.001/0.01), and activation function (ReLU/Tanh).

| Model | Best Validation Accuracy | Best Hyperparameters |
|---|---|---|
| Custom CNN | 94.9% | Adam, dropout 0.2, 512 units, lr 0.001, ReLU |
| **Improved ResNet-50** | **97.42%** | Adam, dropout 0.2, 512 units, lr 0.001, ReLU |

The improved ResNet-50 used a 7×7 initial convolution, four stages of residual blocks (64 → 128 → 256 → 512 filters) with convolutional shortcuts for downsampling, global average pooling, and a sigmoid output for binary classification.

---

## Key Takeaways

Architectural complexity needs to match dataset complexity. MobileNetV2 — despite being a strong general-purpose architecture — underperformed badly on MNIST, while a two-block custom CNN outperformed it with far less compute. On the other hand, for a harder real-world task (malaria detection), the deeper improved ResNet-50 clearly outperformed the simpler custom CNN, justifying its added complexity and tuning cost.

Hyperparameter tuning was decisive across every experiment. Adam consistently outperformed SGD, a learning rate of 0.001 was optimal across both the custom CNN and ResNet-50, and a low dropout rate (0.2) was sufficient regularization in both cases — suggesting these are robust defaults for similarly-scaled image classification tasks.

Overfitting emerged as a recurring risk in the more complex models (Complex CNN Model 2 and the Oxford-IIIT segmentation U-Net), even with batch normalization, dropout, and L1/L2 regularization in place. This reinforces the value of monitoring the training/validation gap rather than relying solely on regularization techniques.

---

## Tools & Methodology

| Tool | Purpose |
|---|---|
| **TensorFlow / Keras** | Model architecture, training, and evaluation |
| **scikit-learn** | IRIS dataset loading, train/test splitting |
| **TensorFlow Datasets** | Malaria and Oxford-IIIT Pet dataset loading |
| **Matplotlib / Seaborn** | Training curves, hyperparameter effect visualization |
| **Grid Search** | Manual nested-loop hyperparameter optimization |

---

## How to Use

```bash
git clone https://github.com/chethan-labs/image-video-classification.git
cd image-video-classification
```

Open the notebooks in order:
1. `Segmentation.ipynb` — synthetic shape segmentation, MobileNetV2-based segmentation, Oxford-IIIT Pet U-Net
2. `Classification_part_1.ipynb` — IRIS baseline, MNIST with MobileNetV2 and custom CNNs
3. `Classification_part_2.ipynb` — Malaria dataset, full hyperparameter grid search, improved ResNet-50

The full methodology, literature review, and detailed results analysis are in `Image_and_Video_Classification_Report.pdf`.

---

## Author

[github.com/chethan-labs](https://github.com/chethan-labs)

MSc Data Science & Analytics, Royal Holloway, University of London — submitted September 2023.
