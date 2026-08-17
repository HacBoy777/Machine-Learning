import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression



# Data = []
# for i in range(1,11):
#     Data.append(i)
# print("Data: ", Data)

data = list(range(1,11))
# print("Data: ", data)

# No Random Sate
train,test = train_test_split(data, test_size=0.3)
# Spilitting with Random State 10
train_10, test_10 = train_test_split(data, test_size=0.3, random_state=10)
# Spilitting with Random State 42
train_42, test_42 = train_test_split(data, test_size=0.3, random_state=42)

print("\nNO Random State:")
print("Train: ", train)
print("Test: ", test)

print("\nWith Random State 10:")
print("Train with Random State 10: ",train_10)
print("Test with Random State 10: ",test_10)

print("\nWith Random State 42:")
print("Train with Random State 42: ",train_42)
print("Test with Random State 42: ",test_42)


train,test = train_test_split(data, test_size=0.3)
train_10, test_10 = train_test_split(data, test_size=0.3, random_state=10)
train_42, test_42 = train_test_split(data, test_size=0.3, random_state=42)
print("\nNO Random State:")
print("Train: ", train)
print("Test: ", test)

print("\nWith Random State 10:")
print("Train with Random State 10: ",train_10)
print("Test with Random State 10: ",test_10)

print("\nWith Random State 42:")
print("Train with Random State 42: ",train_42)
print("Test with Random State 42: ",test_42)


train,test = train_test_split(data, test_size=0.3)
train_10, test_10 = train_test_split(data, test_size=0.3, random_state=10)
train_42, test_42 = train_test_split(data, test_size=0.3, random_state=42)
print("\nNO Random State:")
print("Train: ", train)
print("Test: ", test)

print("\nWith Random State 10:")
print("Train with Random State 10: ",train_10)
print("Test with Random State 10: ",test_10)

print("\nWith Random State 42:")
print("Train with Random State 42: ",train_42)
print("Test with Random State 42: ",test_42)

