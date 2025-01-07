Generative Models (GANs, VAEs)

After transformers, the next step in advancing AI models would be Generative Models, such as Generative Adversarial Networks (GANs) and Variational Autoencoders (VAEs). These models focus on generating new data based on learned patterns, and they are widely used in applications like image synthesis, text generation, and even music composition.

Generative Adversarial Networks (GANs)

Overview
	•	GANs consist of two neural networks: a Generator and a Discriminator.
	•	The Generator creates fake data (e.g., images) that resemble real data.
	•	The Discriminator distinguishes between real and fake data.
	•	Both networks train together in a competitive process, improving each other’s performance over time.

Applications
	1.	Image generation (e.g., creating realistic faces).
	2.	Style transfer (e.g., converting photos to artistic styles).
	3.	Data augmentation (e.g., generating new samples for training).
	4.	Super-resolution (enhancing image quality).

Code Example: Simple GAN for Image Generation

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

Variational Autoencoders (VAEs)

Overview
	•	VAEs are probabilistic models designed to generate data similar to the input data.
	•	They encode input data into a latent space, then sample from this space to reconstruct or generate new data.

Applications
	1.	Image generation and reconstruction.
	2.	Data compression.
	3.	Representation learning.

Code Example: Simple VAE for Image Reconstruction

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
import numpy as np

# VAE parameters
input_dim = 784  # Flattened MNIST images
latent_dim = 2   # Latent space dimension

# Encoder
inputs = Input(shape=(input_dim,))
h = Dense(256, activation='relu')(inputs)
z_mean = Dense(latent_dim)(h)
z_log_var = Dense(latent_dim)(h)

# Sampling function
def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], latent_dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

z = Lambda(sampling)([z_mean, z_log_var])

# Decoder
decoder_h = Dense(256, activation='relu')
decoder_mean = Dense(input_dim, activation='sigmoid')
h_decoded = decoder_h(z)
outputs = decoder_mean(h_decoded)

# VAE model
vae = Model(inputs, outputs)

# Loss function (reconstruction + KL divergence)
reconstruction_loss = binary_crossentropy(inputs, outputs) * input_dim
kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')

# Train VAE
(X_train, _), (_, _) = mnist.load_data()
X_train = X_train.astype('float32') / 255.0
X_train = X_train.reshape(-1, input_dim)
vae.fit(X_train, X_train, epochs=10, batch_size=128)

Next Steps
	1.	Explore Conditional GANs (cGANs):
	•	Generate data conditioned on specific inputs (e.g., generate images of cats or dogs).
	2.	Dive into Applications:
	•	StyleGAN for realistic face generation.
	•	VAEs for anomaly detection in datasets.
	3.	Experiment with Multimodal Generative Models:
	•	Combine text, images, and audio inputs (e.g., DALL-E).

Would you like to explore GAN extensions like StyleGAN or delve deeper into applications of VAEs?