from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
import sklearn
import pandas as pd
from sklearn.metrics import mean_absolute_error
import numpy as np

dt = pd.read_csv("BTH4/Housing_2019.csv", index_col=0)
X = dt.iloc[:, [1, 2, 4, 10]]

Y = dt.price


X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=1.0/3, random_state=100)
len(X_train)  # 1199

tree = DecisionTreeRegressor(random_state=0)

bagging_regtree = BaggingRegressor(
    estimator=tree, n_estimators=10, random_state=42)
bagging_regtree.fit(X_train, y_train)
y_pred = bagging_regtree.predict(X_test)
err = mean_absolute_error(y_test, y_pred)
err
np.sqrt(err)

print(err)
