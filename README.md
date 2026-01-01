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


