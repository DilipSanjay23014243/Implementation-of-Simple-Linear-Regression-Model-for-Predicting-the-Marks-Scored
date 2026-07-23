# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Step-by-step explanation (student friendly)
2. Imports
3. Load dataset
4. Prepare input X and target Y
5. Train/test split
6. Create and train the linear model
7. Predict on test set
8. Plots
9. Evaluation metrics
10. Predict new samples

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: DIlip sanjay M
RegisterNumber:212223240032
*/
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df = pd.read_csv('exp_2_dataset_student_scores.csv')   
print("First 5 rows:\n", df.head(), "\n")
print("Last 5 rows:\n", df.tail(), "\n")
X = df.iloc[:, :-1].values   
Y = df.iloc[:, -1].values    
print("X (features):", X.flatten())
print("Y (targets):", Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1/3, random_state=0)
print("\nTraining samples:", len(X_train), " Testing samples:", len(X_test))
regressor = LinearRegression()
regressor.fit(X_train, Y_train) 
Y_pred = regressor.predict(X_test)
print("\nPredicted values:", np.round(Y_pred, 2))
print("Actual values   :", Y_test)
plt.figure(figsize=(6,4))
plt.scatter(X_train, Y_train, color="orange", label="Training data")
plt.plot(X_train, regressor.predict(X_train), color="red", label="Fitted line")
plt.title("Hours vs Scores (Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()
order = np.argsort(X_test.flatten())
X_test_sorted = X_test.flatten()[order]
Y_test_sorted = Y_test[order]
Y_pred_sorted = Y_pred[order]
plt.figure(figsize=(6,4))
plt.scatter(X_test, Y_test, color="blue", label="Test data")
plt.plot(X_test_sorted, Y_pred_sorted, color="green", label="Predictions")
plt.title("Hours vs Scores (Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.legend()
plt.grid(True)
plt.show()
mae = mean_absolute_error(Y_test, Y_pred)
mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
print("\nMean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print("Root Mean Squared Error (RMSE):", rmse)
new_hours = np.array([[2.5], [8.0]])  
pred_new = regressor.predict(new_hours)
print("\nPredictions for new hours", new_hours.flatten(), "=>", np.round(pred_new,2))



```

## Output:
<img width="718" height="490" alt="Screenshot 2026-07-23 091853" src="https://github.com/user-attachments/assets/d3cfe8bc-be99-4d99-89d8-748303bc685b" />
<img width="681" height="476" alt="Screenshot 2026-07-23 091902" src="https://github.com/user-attachments/assets/4a8e4710-b634-4c38-8350-9998aa5aca42" />
<img width="513" height="103" alt="Screenshot 2026-07-23 091917" src="https://github.com/user-attachments/assets/8eaf9706-1f41-463d-a0aa-9c1c806b213c" />




## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
