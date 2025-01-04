import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Text data
text = "hello world"
chars = sorted(set(text))
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

# Prepare data
seq_length = 3
X = []
y = []
for i in range(len(text) - seq_length):
    X.append([char_to_idx[c] for c in text[i:i + seq_length]])
    y.append(char_to_idx[text[i + seq_length]])
X = np.array(X)
y = np.array(y)

# One-hot encode input and output
vocab_size = len(chars)
X = tf.keras.utils.to_categorical(X, num_classes=vocab_size)
y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

# Build LSTM model
model = Sequential([
    LSTM(50, input_shape=(seq_length, vocab_size)),
    Dense(vocab_size, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy')

# Train the model
model.fit(X, y, epochs=100, verbose=0)

# Generate text
input_seq = "hel"
input_encoded = np.array([[char_to_idx[c] for c in input_seq]])
input_encoded = tf.keras.utils.to_categorical(input_encoded, num_classes=vocab_size)

for _ in range(10):
    prediction = model.predict(input_encoded, verbose=0)
    next_char = idx_to_char[np.argmax(prediction)]
    input_seq += next_char
    input_encoded = np.array([[char_to_idx[c] for c in input_seq[-seq_length:]]])
    input_encoded = tf.keras.utils.to_categorical(input_encoded, num_classes=vocab_size)

print("Generated Text:", input_seq)