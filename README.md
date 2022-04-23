# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1.Import the standard libraries. 
2.Upload the dataset and check for any null or duplicated values using .isnull() and .duplicated() function respectively. 
3.Import LabelEncoder and encode the dataset. 4.Import LogisticRegression from sklearn and apply the model on the dataset. 5.Predict the values of array. 6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn. 7.Apply new unknown values

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: M.Pavan kishore
RegisterNumber:212221230076  
*/
import pandas as pd
data = pd.read_csv("Placement_Data.csv")
data.head()
data1 = data.copy()
data1 = data1.drop(["sl_no","salary"],axis = 1)
data1.head()
data1.isnull().sum()
data1.duplicated().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data1["gender"] = le.fit_transform(data1["gender"])
data1["ssc_b"] = le.fit_transform(data1["ssc_b"])
data1["hsc_b"] = le.fit_transform(data1["hsc_b"])
data1["hsc_s"] = le.fit_transform(data1["hsc_s"])
data1["degree_t"] = le.fit_transform(data1["degree_t"])
data1["workex"] = le.fit_transform(data1["workex"])
data1["specialisation"] = le.fit_transform(data1["specialisation"])
data1["status"] = le.fit_transform(data1["status"])
data1
x = data1.iloc[:,:-1]
x
y = data1["status"]
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 0)
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(solver = "liblinear")
lr.fit(x_train,y_train)
y_pred = lr.predict(x_test)
y_pred
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy
from sklearn.metrics import confusion_matrix
confusion = confusion_matrix(y_test,y_pred)
confusion
from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
classification_report1
lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```


## Output:
HEAD:
![m1](https://user-images.githubusercontent.com/94154941/164910373-f0a45b76-886f-45b7-a695-d761a60f7351.png)
PREDICTED VALUES:
![m2](https://user-images.githubusercontent.com/94154941/164910383-d442ef10-ddd8-4de8-b2c3-5769ba4ea51c.png)
ACCURACY:
![m3](https://user-images.githubusercontent.com/94154941/164910398-14acee4d-8774-4f0c-a32e-1ddb73f3a619.png)
CONFUSION MATRIX:
![m4](https://user-images.githubusercontent.com/94154941/164910410-5cfdaced-0bd7-4094-a2a0-1beecf301af4.png)
CLASSIFICATION REPORT:
![m5](https://user-images.githubusercontent.com/94154941/164910432-8d541594-302f-41b8-8bac-6f4d9a333cf2.png)

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
