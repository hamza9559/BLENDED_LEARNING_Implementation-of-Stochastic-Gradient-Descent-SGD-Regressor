# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Preparation: Load the dataset, define features (Age, Mileage, Horsepower), target (Price), and split into training and testing sets.
2. Feature Scaling: Standardize features using StandardScaler, as SGD is sensitive to the scale of input features.
3. Model Training: Train an SGDRegressor on the training data, and predict prices on the test data.
4. Evaluation & Visualization: Calculate the MSE for model evaluation and create a scatter plot comparing actual vs predicted car prices.

## Program:
```
/*
Program to implement SGD Regressor for linear regression.
Developed by: HAMZA FAROOQUE
RegisterNumber:  212223040054
*/
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Sample data - car attributes (Age, Mileage, Horsepower) and Price
data = {
    'Age': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Mileage': [5000, 15000, 25000, 35000, 45000, 55000, 65000, 75000, 85000, 95000],
    'Horsepower': [100, 110, 120, 130, 140, 150, 160, 170, 180, 190],
    'Price': [20000, 18500, 17500, 16500, 15500, 14500, 13500, 12500, 11500, 10500]
}

# Convert the data to a DataFrame
df = pd.DataFrame(data)

# Feature variables (Age, Mileage, Horsepower)
X = df[['Age', 'Mileage', 'Horsepower']]

# Target variable (Price)
y = df['Price']

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (important for SGD)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Create and train the SGD Regressor
sgd_regressor = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
sgd_regressor.fit(X_train_scaled, y_train)

# Predict car prices using the SGD Regressor
y_pred = sgd_regressor.predict(X_test_scaled)

# Calculate the Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# ----- Visualization: Actual vs Predicted Prices -----
plt.scatter(y_test, y_pred, color='blue', label='Predicted Prices')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2, label='Perfect Prediction Line')

plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Car Prices (SGD Regressor)')
plt.legend()
plt.show()

```

## Output:
```
Mean Squared Error: 22287.578046033515
```

![image](https://github.com/user-attachments/assets/c23c89ac-b42a-4f63-941b-1f4ed37e7ee0)


## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
