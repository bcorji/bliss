

Import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


data = pd.read_csv(‘data.csv’)

x = data.iloc[0]
y = data.iloc[1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

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