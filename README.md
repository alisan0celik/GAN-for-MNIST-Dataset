# Generative Adversarial Network (GAN) for MNIST Dataset

This repository contains the implementation of a Generative Adversarial Network (GAN) using TensorFlow and Keras for generating handwritten digits from the MNIST dataset. GANs are a class of machine learning models designed to generate new data samples that resemble a given dataset.

### Prerequisites

Before running the code, make sure you have the following dependencies installed:

TensorFlow,
NumPy,
Matplotlib

### DataSet

The MNIST dataset is used for training the GAN. It consists of 28x28 grayscale images of handwritten digits (0 through 9). The dataset is loaded and preprocessed to scale pixel values between -1 and 1.


## Model Architecture

### Discriminator

The discriminator is a neural network that classifies whether an input image is real (from the dataset) or fake (generated by the generator). The architecture includes multiple dense layers with LeakyReLU activation functions.


### Generator

The generator is responsible for generating realistic images. It takes random noise as input and generates images that, ideally, are indistinguishable from real ones. The architecture involves dense layers, LeakyReLU activation, and batch normalization.


### GAN

The GAN combines the generator and discriminator into a single model. During training, the generator aims to generate images that fool the discriminator, while the discriminator aims to correctly classify real and generated images.

### Training

The GAN is trained for a specified number of epochs. In each epoch, random noise is used to generate fake images, and a combination of real and fake images is used to train the discriminator and the GAN. Training progress is printed, and generated images are visualized at regular intervals.

## Result

Generated images are displayed at certain intervals during training. You can observe the improvement in image quality as the GAN learns to generate realistic handwritten digits.

## Author
The code was created by Alişan Çelik
