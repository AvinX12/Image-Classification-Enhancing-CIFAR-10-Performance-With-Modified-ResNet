# Image-Classification-Enhancing-CIFAR-10-Performance-With-Modified-ResNet

CS-GY 6953 / ECE-GY 7123 Deep Learning Mini-Project Spring 2024 <br />
New York University, Tandon School of Engineering, Brooklyn, NY 11201, USA <br /> <br />

WORK IN PROGRESS [WIP] <br /> <br />

This repository contains the implementation of a deep learning model for image classification using a **Modified ResNet Architecture**, aimed at enhancing performance on the **CIFAR-10 dataset** under the constraint that model has **no more than 5 million parameters**. <br />

## Overview

The goal of this mini-project is to design and train a convolutional neural network (CNN) model capable of accurately (targeting >90% test accuracy) classifying images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. <br />

## Model Architecture

The model architecture is based on the **ResNet (Residual Network)** framework, which utilizes residual blocks to facilitate training of very deep networks. The ResNet architecture has been modified to the specific requirements of the CIFAR-10 dataset. The modified ResNet architecture consists of multiple convolutional layers, batch normalization layers, and residual blocks. The final layer is a fully connected layer with softmax activation to output class probabilities. <br />

## Training Methodology

The model was trained using the **PyTorch** deep learning framework, using the **CUDA platform** for GPU acceleration. The training process involved multiple epochs, with batch-wise optimization using the Adam optimizer. Data augmentation techniques such as random flips and random crops were employed to increase the diversity of the training dataset and improve generalization performance. <br />

## Performance Evaluation

The performance of the trained model was evaluated on a separate test dataset consisting of previously unseen images from the CIFAR-10 dataset. The evaluation metrics used include test accuracy, test loss, and other relevant performance indicators. <br /> <br />
The final test accuracy achieved by the model was **91.48%** considering just **2,797,610** trainable parameters. <br /> <br />

## Lessons Learned

Throughout the design and training process, several key lessons were learned: <br />
1. Data augmentation techniques play a crucial role in improving model generalization and robustness. <br />
2. Hyperparameter tuning, such as learning rate scheduling and batch size selection, is essential for optimizing model performance. <br /> <br />

## References

// Work In Progress <br />

## Team Members
1. Durga Avinash Kodavalla | dk4852 <br />
2. Priyangshu Pal | pp2833 <br />
3. Ark Pandey | ap8652 <br />
