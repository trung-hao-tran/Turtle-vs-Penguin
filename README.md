# COMP9517 Computer Vision Project

Welcome to the COMP9517 Computer Vision Project repository! This project is part of my COMP9517 coursework and focuses on exploring the power of Convolutional Neural Networks (CNNs) for image classification and object detection tasks. The code lightly touches on CNN concepts yet demonstrates their powerful capabilities, highlighting potential for further research and application.

## Overview

This project includes implementations of two prominent CNN architectures:

1. **VGG16 from Scratch**: A detailed implementation of the VGG16 architecture, demonstrating the foundational concepts of CNNs.
2. **VGG16 from imageNet**: A pre-trained model of VGG16 architecture with custom output layers for both object classification and predict object bounding box
3. **EfficientNetV2**: An advanced, pre-trained model from ImageNet21k, showcasing state-of-the-art performance with efficient computation.

## Project Structure

- `train_annotations.json` and `valid_annotations.json`: JSON files containing the annotations for training and validation datasets.
- `train/`: Directory containing training images.
- `valid/`: Directory containing validation images.
- `vgg16_from_scratch.py`: Script implementing the VGG16 model from scratch.
- `efficientnetv2.py`: Script implementing the EfficientNetV2 model using transfer learning.

## Getting Started

Please checkout the Kaggle competition [Penguins vs Turtles Dataset](https://www.kaggle.com/datasets/abbymorgan/penguins-vs-turtles)

### Prerequisites

- Python 3.x
- TensorFlow
- OpenCV
- Pandas
- NumPy
- Matplotlib
- scikit-learn