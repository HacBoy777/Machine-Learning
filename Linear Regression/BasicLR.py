from sklearn import linear_model

# Regression model
reg = linear_model.LinearRegression()

## Supervised Learning :-
# As we are taking features and labels both
# Data
# X ==> Features ==> 2D Array 
# y ==> Labels/Target ==> 1D Array
X = [[1], [2], [3],[4],[5],[6]]
y = [2,2.5,4.5,3,5,4.7]

# Training the model
reg.fit(X, y)

# Predicting the values
result = reg.predict([[5.5]])
print("Predicted value for input 5.5 is:", result)