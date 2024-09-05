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
![364298081-cf0a7cb3-f086-4f9c-8360-ffbcdf402808](https://github.com/user-attachments/assets/b362db54-16e5-4e73-8cd7-f14990fe7891)


~~~python
df.info()
~~~
![364298410-8acf2aaf-8108-4fe8-bc05-2d79d2ffe7f9](https://github.com/user-attachments/assets/af782379-e628-4ebe-874b-37988bdfdc1f)

~~~python
X=df.drop(columns=['AveOccup','target'])
X.info()
~~~
![364298746-a2305206-cfea-4d68-b0ae-576259660220](https://github.com/user-attachments/assets/85a38715-1b33-4afa-8fb3-ede9d43a7224)

~~~python
Y=df[['AveOccup','target']]
Y.info()
~~~
![364299224-226d87cd-cf5b-4137-bdaf-8ae683d9f936](https://github.com/user-attachments/assets/c1a08c8b-6d42-466b-9ae7-13ffccfb6882)

~~~python
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)
X.head()
~~~
![364299527-8c472cc7-88e2-434a-8932-cb2eaa258199](https://github.com/user-attachments/assets/d03b8fad-f433-472b-a4cb-ec6b7b64bd22)

~~~python
scaler_X=StandardScaler()
scaler_Y=StandardScaler()
X_train=scaler_X.fit_transform(X_train)
X_test=scaler_X.transform(X_test)
Y_train=scaler_Y.fit_transform(Y_train)
Y_test=scaler_Y.transform(Y_test)
print(X_train)
~~~
![364299952-74067f93-ba6f-4a57-bb0e-aef7c777cf1d](https://github.com/user-attachments/assets/1d3ef95f-afbe-4c90-8a69-7483ce295cea)

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
![364300532-00db5521-9379-4fc9-820d-e491fab3ac68](https://github.com/user-attachments/assets/5e99735c-08d6-48fb-8224-25aacc055cbe)




## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
