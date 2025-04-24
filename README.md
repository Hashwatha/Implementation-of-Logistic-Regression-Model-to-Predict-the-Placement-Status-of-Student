# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start
2. Load the California Housing dataset and select the first 3 features as input (X) and target variables (Y) (including the target price and another feature).
3. Split the data into training and testing sets, then scale (standardize) both the input features and target variables.
4. Train a multi-output regression model using Stochastic Gradient Descent (SGD) on the training data.
5. Make predictions on the test data, inverse transform the predictions, calculate the Mean Squared Error, and print the results.
6. Stop
## Program // Output::
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Hashwatha M
RegisterNumber: 212223240051
*/
```
```
import pandas as pd
df=pd.read_csv('Placement_Data.csv')
df.head()
```
![image](https://github.com/user-attachments/assets/3724ad43-c9f1-449e-bf4a-1342a2505ab9)
```
data1 = df.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
```
![image](https://github.com/user-attachments/assets/98a46fe3-cd00-4199-a23b-e3d1ea4ba678)
```
data1.isnull().sum()
```
![image](https://github.com/user-attachments/assets/d99f2a8c-3f8e-4e87-8ec7-63dcdebfb7c5)
```
data1.duplicated().sum()
```
![image](https://github.com/user-attachments/assets/284cc264-bed5-4275-9d6c-782b4f70a296)
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_p"] = le.fit_transform(data1["degree_p"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["etest_p"] = le.fit_transform(data1["etest_p"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
```
![image](https://github.com/user-attachments/assets/bfa44a6a-a322-49fc-a97c-6b7b626a391d)
```
x = data1.iloc[:,:-1]
x
```
![image](https://github.com/user-attachments/assets/26c53084-b577-48f0-9c84-c0736746a7ae)
```
y = data1["status"]
y
```
![image](https://github.com/user-attachments/assets/f3490de0-121f-4d25-ae42-f12f01bb87bc)
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 1/3,random_state = 0)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
```
![image](https://github.com/user-attachments/assets/77df6310-5d7e-4b41-a78e-91dea7803e76)
```
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/61d0e084-81f9-4958-b65b-3d7ae55e6e64)
```
from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion
```
![image](https://github.com/user-attachments/assets/341108bd-9e3b-4a74-9416-6696db70ea11)
```
from sklearn.metrics import classification_report
classification_report=classification_report(y_test,y_pred)
print(classification_report)
print("Name:Hashwatha M")
print("Reg no:212223240051")
```
![image](https://github.com/user-attachments/assets/28340ac3-a30a-4c76-a4dc-34955afa72a4)

```
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```
![image](https://github.com/user-attachments/assets/be9403c2-2f0b-40d0-8a72-8a687ba401d3)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
