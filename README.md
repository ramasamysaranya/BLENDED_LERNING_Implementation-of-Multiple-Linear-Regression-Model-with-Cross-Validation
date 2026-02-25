# BLENDED_LERNING
# Implementation-of-Multiple-Linear-Regression-Model-with-Cross-Validation-for-Predicting-Car-Prices

## AIM:
To write a program to predict the price of cars using a multiple linear regression model and evaluate the model performance using cross-validation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the car price dataset, select relevant numerical features (enginesize, horsepower, citympg, highwaympg) as input variables, and set price as the target variable. Split the data into training and testing sets.
2. Apply standardization to the training features using StandardScaler and transform the testing features using the same scaler to ensure consistent feature scaling.
3. Train a Linear Regression model using the scaled training data, predict prices for the test data, and evaluate model performance using MSE, RMSE, and R-squared metrics along with model coefficients.
4. Check linearity using actual vs predicted plots, test independence of errors using the Durbin–Watson statistic, assess homoscedasticity through residual plots, and verify normality of residuals using histogram and Q–Q plots.

## Program:
```
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

#1.Load and prepare data
data = pd.read_csv('CarPrice_Assignment.csv')

#Simple preprocessing
data=data.drop(['car_ID','CarName'],axis=1)
data=pd.get_dummies(data,drop_first=True)

#2.split data
X=data.drop('price',axis=1)
y=data['price']
X_train, X_test ,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#4. Evaluate with cross-validation(simple version)
print("Name: SARANYA R")
print("Reg. No: 212225040384")
print("\n=== Cross-Validation ===")
cv_scores = cross_val_score(model,X,y,cv=5)
print("Fold R2 scores:",[f"{score:.4f}" for score in cv_scores])
print(f"Average R2: {cv_scores.mean():.4f}")

#5.Test set evaluation
y_pred = model.predict(X_test)
print("\n Test Set Perfoemance")
print(f"MSE: {mean_squared_error(y_test, y_pred):.4f}")
print(f"R2: {r2_score(y_test,y_pred):.4f}")

#6.visualization
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([y.min(), y.max()],[y.min(),y.max()],'r--')
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Prices")
plt.grid(True)
plt.show()
```

## Output:
<img width="752" height="148" alt="image" src="https://github.com/user-attachments/assets/408d0c92-9ba5-4d34-a0b2-2730c6c2137d" />
<img width="362" height="95" alt="image" src="https://github.com/user-attachments/assets/904234e8-9848-42c9-9894-e8f7603bf8ba" />
<img width="1136" height="694" alt="image" src="https://github.com/user-attachments/assets/da497409-fe5c-4be8-b20b-e4205d660262" />



## Result:
Thus, the program to implement the multiple linear regression model with cross-validation for predicting car prices is written and verified using Python programming.
