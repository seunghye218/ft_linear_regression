import sys
import numpy as np
import pandas as pd

if len(sys.argv) != 2:
    print(f'usage: python3 {sys.argv[0]} [data file]')
    sys.exit(0)

with open('theta.csv', 'r') as file:
    lines = file.readlines()

theta = [float(i) for i in lines[0].split(',')]

# 1. Importing data
data = pd.read_csv(sys.argv[1])

# 2. Data preprocessing
X = data['km'].values
y = data['price'].values

predic_y = [theta[0] + theta[1] * X]
r_squared = 1 - (np.sum((y - predic_y)**2)) / np.sum((y - [np.mean(y) for i in range(len(y))])**2)

print(f"R-squared: {r_squared}")
