import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

# Generate sine wave data
def generate_sine_wave_data(seq_length=50, num_samples=1000):
    X = []
    y = []
    for _ in range(num_samples):
        start = np.random.rand()
        x = np.linspace(start, start + seq_length * 0.01, seq_length)
        sine_wave = np.sin(2 * np.pi * x)
        X.append(sine_wave[:-1])  # Input sequence
        y.append(sine_wave[1:])  # Target sequence
    return np.array(X), np.array(y)

# Prepare data
seq_length = 50
X, y = generate_sine_wave_data(seq_length=seq_length)
X = X[..., np.newaxis]  # Add channel dimension
y = y[..., np.newaxis]

# Build GRU model
model = Sequential([
    GRU(10, activation='tanh', input_shape=(seq_length - 1, 1)),
    Dense(1)  # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X, y, epochs=10, batch_size=32)

# Test the model
test_input = X[0:1]
predicted_output = model.predict(test_input)
print("Predicted:", predicted_output.flatten())
print("Actual:", y[0].flatten())