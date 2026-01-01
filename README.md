# CNN Image Classification on CIFAR-10 (PyTorch)

A simple end-to-end Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset to classify natural images into 10 categories.

## Project Overview

This project demonstrates a complete deep learning pipeline in PyTorch:
- Data loading and preprocessing with `torchvision`
- Implementing a simple CNN architecture
- Training and evaluating the model
- Analyzing results and discussing limitations

The focus is on understanding the pipeline rather than achieving state-of-the-art performance.

## Dataset

**CIFAR-10**: 60,000 color images (32×32 pixels) in 10 classes:
- airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck
- Training set: 50,000 images
- Test set: 10,000 images

*Note: For this demonstration, we train on a 10,000-image subset to reduce computational time.*

## Architecture

A small CNN with 3 convolutional layers and 2 fully connected layers:

```
Input (3×32×32)
  ↓
Conv2d(3→32) + ReLU + MaxPool2d
  ↓
Conv2d(32→64) + ReLU + MaxPool2d
  ↓
Conv2d(64→128) + ReLU + MaxPool2d
  ↓
Flatten + Linear(128×4×4 → 256) + ReLU
  ↓
Linear(256 → 10)
  ↓
Output (10 classes)
```

## Results

Trained for **2 epochs** on a **10,000-image subset** of CIFAR-10:

| Metric | Train | Test |
|--------|-------|------|
| Loss   | 1.31  | 1.25 |
| Accuracy | 52.4% | 54.5% |

*Note: This is a demonstration setting (limited data, few epochs). With full data and more epochs, accuracy would be significantly higher.*

## Tech Stack

- **Python 3.8+**
- **PyTorch** - Deep learning framework
- **torchvision** - Dataset and transforms
- **NumPy, Matplotlib** - Analysis and visualization

## Installation

```bash
# Clone the repo
git clone https://github.com/sokhna-ai/cnn-cifar10-pytorch.git
cd cnn-cifar10-pytorch

# Install dependencies
pip install -r requirements.txt
```

## Usage

Open `cnn_cifar10.ipynb` in Jupyter Notebook and run all cells.

The notebook will:
1. Download CIFAR-10 (if not already cached)
2. Create a 10,000-image training subset
3. Instantiate the CNN model
4. Train for 2 epochs
5. Evaluate on the test set
6. Display results

## Key Observations

- **What the model learned**: Even with limited data and training, the model achieves ~54% test accuracy (vs ~10% random baseline), showing it learned meaningful features.
- **Generalization gap**: The test accuracy (54%) is close to training accuracy (52%), indicating the model isn't severely overfitting despite the small dataset.
- **Room for improvement**: A full training setup (all 50k images, 20+ epochs, data augmentation, learning rate scheduling) would achieve >70% accuracy.

## Files

- `cnn_cifar10.ipynb` - Main notebook with complete pipeline
- `README.md` - This file
- `requirements.txt` - Python dependencies

## Limitations & Future Work

- **Small training set**: Using only 10k images instead of full 50k to speed up training
- **Few epochs**: Stopping at 2 epochs instead of convergence
- **Simple architecture**: Using a small CNN instead of modern architectures (ResNet, EfficientNet)
- **No hyperparameter tuning**: Learning rate, batch size, and other hyperparameters not optimized

Future improvements could include:
- Training on the full CIFAR-10 dataset
- Implementing ResNet or other modern architectures
- Using learning rate scheduling and more aggressive augmentation
- Transfer learning from pretrained models (e.g., ImageNet)

## References

- [CIFAR-10 Dataset](https://www.cs.toronto.edu/~kriz/cifar.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [PyTorch Vision Models](https://pytorch.org/vision/stable/models.html)
