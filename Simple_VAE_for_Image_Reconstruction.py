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