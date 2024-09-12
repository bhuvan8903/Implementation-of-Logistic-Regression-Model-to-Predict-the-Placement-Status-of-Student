# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student
## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook=
## Algorithm
```
1.Import the required packages.
2.Print the present data and placement data and salary data.
3.Using logistic regression find the predicted values of accuracy confusion matrices.
4.Display the results.
```
## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: MUKESH KUMAR S
RegisterNumber:  212223240099
```
```
import pandas as pd
data=pd.read_csv("Placement_Data.csv")
data.head()
data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"])
data1["status"]=le.fit_transform(data1["status"])
data1.head()
x=data1.iloc[:,:-1]
print(x) #allocate the -1 column for x
y=data1["status"]
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver="liblinear")
#classification
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
print(y_pred)
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
print(accuracy)
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
print(confusion)
from sklearn.metrics import classification_report
classification_report1=classification_report(y_test,y_pred)
print(classification_report1)
print(lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]]))
```
## Output:
## Placement Data:
![image](https://github.com/user-attachments/assets/9fb5c35c-5fa0-4608-b7d1-2f68d7f96bf9)

## Checking the null() function:
![image](https://github.com/user-attachments/assets/f6c8f989-f938-4f0a-b158-6ff4f659472f)

## Data Duplicate:
![image](https://github.com/user-attachments/assets/d0a120a5-da8e-4871-b515-d654f21baef6)

## Print Data:
![image](https://github.com/user-attachments/assets/72d37b73-160e-40a8-a810-d6093b710525)

## Y_prediction array:
![image](https://github.com/user-attachments/assets/a9ee28b7-0413-46bd-8e31-aae80669fc67)

## Accuracy value:
![image](https://github.com/user-attachments/assets/94ecdcee-a65b-4770-90ef-ad3e4c1ceb14)

## Confusion array:
![image](https://github.com/user-attachments/assets/b8380f01-104d-4973-a9a0-aae5f574ffd6)

## Classification Report:
![image](https://github.com/user-attachments/assets/ba0f8e4b-d850-4918-b214-0e4df5e7ead5)

## Prediction of LR:
![image](https://github.com/user-attachments/assets/114ee6fa-c561-4a90-a02e-343a7f16461f)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
