import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Generate sine wave data
def generate_sine_wave_data(seq_length, num_samples):
    X = []
    y = []
    for _ in range(num_samples):
        start = np.random.rand() * 2 * np.pi
        x = np.sin(np.linspace(start, start + seq_length * 0.1, seq_length))
        X.append(x[:-1])
        y.append(x[-1])
    return np.array(X), np.array(y)

seq_length = 50
num_samples = 1000
X, y = generate_sine_wave_data(seq_length, num_samples)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape the data to fit the RNN input requirements
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# Create an RNN model
model = Sequential([
    SimpleRNN(50, activation='tanh', input_shape=(seq_length - 1, 1)),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Evaluate the model
test_loss = model.evaluate(X_test, y_test)
print(f"Test loss: {test_loss}")

# Make predictions
y_pred = model.predict(X_test)

# Plot the results
plt.plot(y_test, label='True')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.show()