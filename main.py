# The code was created by Alişan Çelik
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LeakyReLU, BatchNormalization, Reshape, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist

# Load dataset
(X_train, _), (_, _) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)

# Model parameters for GAN
random_dim = 100

# Diskriminator model
def build_discriminator():
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28, 1)))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return model

# Generator model
def build_generator():
    model = Sequential()
    model.add(Dense(256, input_dim=random_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(28*28*1, activation='tanh'))
    model.add(Reshape((28, 28, 1)))
    return model

# GAN model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))
    return model

# Create model
discriminator = build_discriminator()
generator = build_generator()
discriminator.trainable = True
gan = build_gan(generator, discriminator)

# Training
epochs = 50001
batch_size = 128

for e in range(epochs):
    noise = np.random.normal(0, 1, size=[batch_size, random_dim])
    generated_images = generator.predict(noise)
    image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]

    X = np.concatenate([image_batch, generated_images])
    y_dis = np.zeros(2*batch_size)
    y_dis[:batch_size] = 0.9

    d_loss = discriminator.train_on_batch(X, y_dis)

    noise = np.random.normal(0, 1, size=[batch_size, random_dim])
    y_gen = np.ones(batch_size)
    g_loss = gan.train_on_batch(noise, y_gen)

    if e % 500 == 0:
        print(f"Epoch {e}, D Loss: {d_loss}, G Loss: {g_loss}")

    # Show images produced between specific epochs
    if e % 1000 == 0:
        noise = np.random.normal(0, 1, size=[100, random_dim])
        generated_images = generator.predict(noise)
        generated_images = generated_images.reshape(100, 28, 28)

        plt.figure(figsize=(10, 10))
        for i in range(generated_images.shape[0]):
            plt.subplot(10, 10, i+1)
            plt.imshow(generated_images[i], interpolation='nearest', cmap='gray_r')
            plt.axis('off')
        plt.tight_layout()
        plt.show()
# The code was created by Alişan Çelik
