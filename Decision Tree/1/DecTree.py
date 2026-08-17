## DATA PREPROCESSING  ==> STARTS
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df=pd.read_csv("data\\salaries.csv")
# print(df[:5])
# print(df.shape)         ## (16, 4)

X=df[["company","job","degree"]]
y=df[["salary_more_then_100k"]]

# print(X[:5])
# print(y[:5])

lbl_company= LabelEncoder()
lbl_job=LabelEncoder()
lbl_degree=LabelEncoder()

companyin= lbl_company.fit_transform(X['company'])
# print(companyin)                      ## [2 2 2 2 2 2 0 0 0 0 1 1 1 1 1 1]
jobin = lbl_job.fit_transform(X['job'])
# print(jobin)                            ## [2 2 0 0 1 1 2 1 0 0 2 2 0 0 1 1]
degreein=lbl_degree.fit_transform(X['degree'])
# print(degreein)                         ## [0 1 0 1 0 1 1 0 0 1 0 1 0 1 0 1]


X['companyin'] = companyin
X['jobin'] = jobin
X['degreein'] = degreein
# print(X.head())

new_X=X.drop(['company','degree','job'],axis='columns')
# print(new_X.head())
# print(new_X.columns)
# print(X.columns)
print(new_X)

##  DATA PREPROCESSING ==> ENDS

## TRAINING CLASSIFIER BEGINS...!!!
from sklearn import tree
model=tree.DecisionTreeClassifier()
model.fit(new_X,y)
# print(model.score(new_X,y))          ## 1.0
print(model.predict([[1,0,0]]))

## TRAINING CLASSIFIER ENDS...!!