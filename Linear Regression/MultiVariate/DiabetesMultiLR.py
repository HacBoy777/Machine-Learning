 # import pandas as pd
# import numpy as np
# import matplotlib.pyplot  as plt
# from sklearn import linear_model      ## from sklearn import linear_model, datasets
# from sklearn import datasets
# from sklearn.metrics import mean_squared_error

# # Load the diabetes dataset
# diabetes = datasets.load_diabetes()
# ##now take Dia_x = 9 which shape 442x9 and dia_y = 1 which shape 442x1 with use of np.newaxis
# diabetes_X = diabetes.data[:, :9]
# print("Shape of Dia_x:", diabetes_X.shape)  # Should be (442, 9)
# diabetes_y = diabetes.target[:, np.newaxis]
# print("Shape of dia_y:", diabetes_y.shape)  # Should be (442, 1)

# diabetes_X_train = diabetes_X[:-42]

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn import datasets,linear_model
diabetes=datasets.load_diabetes()

diabetes_x=diabetes.data[:,:9]
# print(diabetes_x.shape)
diabetes_y=diabetes.data[:,9:10]
# print(diabetes_y.shape)

diabetes_x_train=diabetes_x[:-42]
# print(diabetes_x_train.shape)##(400,9)
diabetes_x_test=diabetes_x[-42:]
# print(diabetes_x_test.shape)##(42,9)
diabetes_y_train=diabetes_y[:-42]
# print(diabetes_y_train.shape)##(400,1)
diabetes_y_test=diabetes_y[-42:]
# print(diabetes_y_test.shape) #(42,1)

lr_model= linear_model.LinearRegression()
lr_model.fit(diabetes_x_train,diabetes_y_train)
pred_y_diabetes=lr_model.predict(diabetes_x_test)
print(pred_y_diabetes)
print(pred_y_diabetes.shape)

print("mean squared error",mean_squared_error(diabetes_y_test,pred_y_diabetes))