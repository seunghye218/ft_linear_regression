import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 0. Receive as data file argument
if len(sys.argv) != 2:
    print(f'usage: python3 {sys.argv[0]} [data file]')
    sys.exit(0)

# 1. Importing data
data = pd.read_csv(sys.argv[1])

# 2. Data preprocessing
X = data['km'].values
y = data['price'].values

# 3. Initialize model parameters
learning_rate = 0.001
iterations = 10000
m = len(y)
theta = np.zeros(2)

# Data normalization
X_mean = np.mean(X)
X_std = np.std(X)
X = (X - X_mean) / X_std

# 4. Implementing the gradient descent algorithm
for _ in range(iterations):
    # Calculating predicted values
    y_pred = theta[0] + theta[1] * X

    # Calculating error
    error = y_pred - y

    # Updating parameters
    gradient_0 = np.sum(error) / m
    gradient_1 = np.sum(error * X) / m
    theta[0] -= learning_rate * gradient_0
    theta[1] -= learning_rate * gradient_1

    # Code for Debugging
    # if _ % 100 == 0:
    #    print(f'gradient_0 : {gradient_0}')
    #    print(f'gradient_1: {gradient_1}')
    #    print(f'epoch: {_}, theta0: {theta[0]}, theta1: {theta[1]}')

# Restore to original scale with inverse normalization
theta[0] = theta[0] - theta[1] * X_mean / X_std
theta[1] = theta[1] / X_std
X = X*X_std + X_mean

# Restored theta0 and theta1 outputs
print(f"theta0: {theta[0]}")
print(f"theta1: {theta[1]}")
with open('theta.csv', 'w') as file:
    file.write(f"{theta[0]},{theta[1]}\n")
print("Save the values of theta0 and theta1 to the theta.csv file.")

# 5. Visualize results
plt.scatter(X, y, label='Real Price')
plt.plot(X, (theta[0] + theta[1] * X),
        color='red', label='Predict Price')
plt.xlabel('(km)')
plt.ylabel('Price')
plt.legend()
plt.title('Predict Car Price')
plt.show()
