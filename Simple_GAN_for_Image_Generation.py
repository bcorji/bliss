import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten, LeakyReLU
from tensorflow.keras.datasets import mnist
import numpy as np

# Load MNIST dataset
(X_train, _), (_, _) = mnist.load_data()
X_train = (X_train / 127.5) - 1.0  # Normalize to [-1, 1]
X_train = X_train.reshape(-1, 28 * 28)

# Generator model
def build_generator():
    model = Sequential([
        Dense(128, input_dim=100),
        LeakyReLU(alpha=0.2),
        Dense(784, activation='tanh'),  # Output size matches the flattened MNIST image
        Reshape((28, 28, 1))  # Reshape to image dimensions
    ])
    return model

# Discriminator model
def build_discriminator():
    model = Sequential([
        Flatten(input_shape=(28, 28, 1)),
        Dense(128),
        LeakyReLU(alpha=0.2),
        Dense(1, activation='sigmoid')  # Output probability (real or fake)
    ])
    return model

# Build and compile GAN
generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# GAN: Combine generator and discriminator
discriminator.trainable = False
gan_input = tf.keras.Input(shape=(100,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = tf.keras.Model(gan_input, gan_output)
gan.compile(optimizer='adam', loss='binary_crossentropy')

# Training loop
def train_gan(epochs=10000, batch_size=128):
    for epoch in range(epochs):
        # Train discriminator
        real_images = X_train[np.random.randint(0, X_train.shape[0], batch_size)]
        fake_images = generator.predict(np.random.randn(batch_size, 100))
        X = np.concatenate([real_images, fake_images])
        y = np.concatenate([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])
        discriminator.train_on_batch(X, y)

        # Train generator
        noise = np.random.randn(batch_size, 100)
        y_fake = np.ones((batch_size, 1))  # Trick discriminator into thinking generated images are real
        gan.train_on_batch(noise, y_fake)

        if epoch % 1000 == 0:
            print(f"Epoch {epoch} completed.")

train_gan()