# Facial Emotion Recognition using CNN

This repository contains the implementation of a Convolutional Neural Network (CNN) for facial emotion recognition. The model leverages deep learning techniques to identify and classify different human emotions from facial images.

## 1. Dataset
Our models are trained on the fer2013_csv.csv dataset, which includes pixel-level representations of facial expressions labeled across seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
Dataset Source: [Kaggle](https://www.kaggle.com/datasets/ahmedmoorsy/facial-expression/data)


## 2. Model Architecture
### 2.1 ViT Model
The Vision Transformer (ViT) employed for facial emotion recognition utilizes a structured approach by dividing images into patches and embedding each patch. The architecture comprises a patch embedding layer, multiple transformer encoder blocks, and a classification head. Each encoder block integrates a multi-head self-attention mechanism and a feedforward network, with layer normalization applied before and after the attention process.

<img src="/VisionTransformer(ViT).png">

### 2.2 CNN Model

The CNN model uses a Residual Network (ResNet) architecture with several Residual Blocks, including convolutional layers with ReLU activation and batch normalization. The network architecture involves initial convolutional layers followed by multiple Residual Blocks, average pooling, and a final dense layer with softmax activation for classification.

<img src="/CNNsModel.png">
