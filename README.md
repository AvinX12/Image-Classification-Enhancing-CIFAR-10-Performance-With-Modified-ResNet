# Image-Classification-Enhancing-CIFAR-10-Performance-With-Modified-ResNet

CS-GY 6953 / ECE-GY 7123 Deep Learning Mini-Project Spring 2024 <br />
New York University (NYU) Tandon School of Engineering <br /> <br />

Mini-Project still in WORK IN PROGRESS ! <br /> <br />

This repository contains a deep learning model implementation for image classification using the **Modified ResNet Architecture**, which aims to improve performance on a **CIFAR-10** dataset under the constraint that the model **does not exceed 5 million trainable parameters** <br />

## Overview

The objective of this mini-project is to develop and train a convolutional neural network (CNN) model to classify images from the CIFAR-10 dataset accurately (targeting > 90% test accuracy). The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. <br />

## Model Architecture

The model architecture is based on the ResNet (Residual Network) framework, which uses residual blocks to facilitate training of very deep networks. The ResNet architecture has been modified to the specific requirements of the CIFAR-10 dataset. The modified ResNet architecture consists of multiple convolutional layers, batch normalization layers, and residual blocks. The final layer is a fully connected layer with softmax activation to output class probabilities. <br />

## Training Methodology

The model was trained using the **PyTorch** deep learning framework, using the CUDA platform. The training process involved multiple epochs, with batch-wise optimization using the Adam optimizer. Data augmentation techniques such as random flips and random crops were employed to increase the diversity of the training dataset and improve generalization performance. <br />

## Performance Evaluation

The performance of the trained model was evaluated on a separate test dataset consisting of previously unseen images from the CIFAR-10 dataset. The evaluation metrics used were test accuracy, and test loss. <br /> <br />
The final test accuracy achieved by the model was **91.48%** considering just **2,797,610** trainable parameters. <br /> <br />

## References

1. Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. [click here](https://www.cs.toronto.edu/~kriz/cifar.html)
2. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. CoRR, abs/1512.03385. [click here](https://doi.org/10.48550/arXiv.1512.03385)
<br /> <br />

## Team Members
1. Durga Avinash Kodavalla | dk4852 <br />
2. Priyangshu Pal | pp2833 <br />
3. Ark Pandey | ap8652 <br />
