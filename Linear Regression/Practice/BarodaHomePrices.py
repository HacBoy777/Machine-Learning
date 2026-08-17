import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


df = pd.read_csv('Data\\barodahomeprices1.csv')
# print(df)
# print(df.shape)

dumies = pd.get_dummies(df.town)
# print(dumies)

df_dumies = pd.concat([df, dumies], axis=1)
# print(df_dumies)

df_dumies.drop(['town'], axis=1, inplace=True)
# print(df_dumies)

df_dumies.drop(['Gotri'], axis=1, inplace=True)
# print(df_dumies)

X = df_dumies[["area","Bhayli","Karelibaug"]]
y = df_dumies['price']

homesize = int(input("Enter home size in sq ft: "))
areaname = input("Enter area name: \nB. Bhayli\nK. Karelibaug\nG. Gotri\n")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lr_model = LinearRegression()
lr_model.fit(X_train,y_train)

B=0
K=0
G=0

if areaname == 'B' or areaname == 'b':
    B = 1
elif areaname == 'K' or areaname == 'k':
    K = 1
else:
    G = 1

print(lr_model.predict([[homesize, B, K]]))