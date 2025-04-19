# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the standard libraries.

2. Upload the dataset and check for any null values using .isnull() function.

3. Import LabelEncoder and encode the dataset.

4. Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

5. Predict the values of arrays.

6. Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

7. Predict the values of array.

8. Apply to new unknown values.

## Program:
```

Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by  : MOHAMED RASHITH S
RegisterNumber: 212223243003

```
```py
import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]
y.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred

from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])
```

## Output:
## Head value:
![image](https://github.com/user-attachments/assets/fb5d4b4e-6df9-45a4-9704-6b50c52d3711)
## Converting string literals to numerical values using label encoder:
![image](https://github.com/user-attachments/assets/f0367be4-65c1-43d2-9f07-60f103226fdf)
## MEAN SQUARED ERROR:
![image](https://github.com/user-attachments/assets/28c41635-6e3d-40a1-97ff-7b3538e4797f)

## R2 (Variance):
![image](https://github.com/user-attachments/assets/d9956938-4854-456f-81b1-a2d6ffd6c523)

### predicted value:
![image](https://github.com/user-attachments/assets/3f0f50f6-da47-41e6-a759-a4ec2f52a404)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
