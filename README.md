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
```c
/*
Program to implement SGD Regressor for linear regression.
Developed by: HAMZA FAROOQUE
RegisterNumber: 212223040054
*/
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = 'encoded_car_data.csv'
df = pd.read_csv(file_path)

# Select relevant features and target variable
X = df.drop(columns=['price'])  # All columns except 'price'
y = df['price']  # Target variable

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the SGD Regressor
sgd_model = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)  # Default settings
sgd_model.fit(X_train, y_train)

# Predictions on test set
y_pred = sgd_model.predict(X_test)

# Evaluate the model
print("Model Performance:")
print("Mean Squared Error (MSE):", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# Print model coefficients
print("\nModel Coefficients:")
print("Coefficients:", sgd_model.coef_)
print("Intercept:", sgd_model.intercept_)

```

## Output:

![image](https://github.com/user-attachments/assets/bba4a276-a928-4926-81b0-2d6ff59d878b)



## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
