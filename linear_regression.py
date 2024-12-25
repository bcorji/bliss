import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 1) * 10  # Features (random numbers between 0 and 10)
y = 2.5 * X + np.random.randn(100, 1) * 2  # Target variable with noise

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict using the model
y_pred = model.predict(X_test)

# Visualize the results
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', label='Regression Line')
plt.xlabel('Feature (X)')
plt.ylabel('Target (y)')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()

# Print the model parameters
print(f"Weight (w): {model.coef_[0][0]}")
print(f"Bias (b): {model.intercept_[0]}")
print(f"Model Score (R^2): {model.score(X_test, y_test):.2f}")
