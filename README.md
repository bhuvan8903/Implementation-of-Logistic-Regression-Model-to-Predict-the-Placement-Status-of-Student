# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2.Print the placement data and salary data.
3.Find the null and duplicate values.
4.Using logistic regression find the predicted values of accuracy , confusion matrices.
5.Display the results. 

## Program:

#### Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
#### Developed by: BHUVANESHWARAN H
#### RegisterNumber:  212223240018


~~~python
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
~~~
~~~python
data=fetch_california_housing()
df=pd.DataFrame(data.data,columns=data.feature_names)
df['target']=data.target
print(df.head())
~~~
![Screenshot 2024-09-04 141320](https://github.com/user-attachments/assets/cf0a7cb3-f086-4f9c-8360-ffbcdf402808)
~~~python
df.info()
~~~
![Screenshot 2024-09-04 141326](https://github.com/user-attachments/assets/8acf2aaf-8108-4fe8-bc05-2d79d2ffe7f9)
~~~python
X=df.drop(columns=['AveOccup','target'])
X.info()
~~~
![Screenshot 2024-09-04 141334](https://github.com/user-attachments/assets/a2305206-cfea-4d68-b0ae-576259660220)
~~~python
Y=df[['AveOccup','target']]
Y.info()
~~~
![Screenshot 2024-09-04 141342](https://github.com/user-attachments/assets/226d87cd-cf5b-4137-bdaf-8ae683d9f936)
~~~python
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
X.head()
~~~
![Screenshot 2024-09-04 141402](https://github.com/user-attachments/assets/8c472cc7-88e2-434a-8932-cb2eaa258199)
~~~python
scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)
print(X_train)
~~~
![Screenshot 2024-09-04 141409](https://github.com/user-attachments/assets/74067f93-ba6f-4a57-bb0e-aef7c777cf1d)
~~~python
sgd=SGDRegressor(max_iter=1000,tol=1e-3)
multi_output_sgd=MultiOutputRegressor(sgd)
multi_output_sgd.fit(X_train,Y_train)
Y_pred=multi_output_sgd.predict(X_test)
Y_pred = scaler_Y.inverse_transform(Y_pred)
Y_test = scaler_Y.inverse_transform(Y_test)
mse = mean_squared_error(Y_test, Y_pred)
print("Mean Squared Error:", mse)
print("\nPredictions:\n", Y_pred[:5])
~~~

## Output:
![Screenshot 2024-09-04 141416](https://github.com/user-attachments/assets/00db5521-9379-4fc9-820d-e491fab3ac68)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
