# doc du lieu tu file housing
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
import matplotlib.pyplot as plt
import sklearn
import pandas as pd
import numpy as np

dt = pd.read_csv("Housing_2019.csv", index_col=0)
dt.iloc[2:4, ]
X = dt.iloc[:, [1, 2, 3, 4, 10]]
X.iloc[1:5, ]
y = dt.price
plt.scatter(dt.lotsize, dt.price)
plt.show()

# huan luyen mo hinh

lm = linear_model.LinearRegression()
lm.fit(X[0:520], y[0:520])

print(lm.intercept_)

print(lm.coef_)

# du bao gia nha cho 20 phan tu cuoi trong tap du lieu

y = dt.price
y_test = y[-20:]
X_test = X[-20:]
y_pred = lm.predict(X_test)

# so sanh gia tri thuc te va gia tri du bao

y_pred
y_test


err = mean_squared_error(y_test, y_pred)
err
rmse_err = np.sqrt(err)
round(rmse_err, 3)
