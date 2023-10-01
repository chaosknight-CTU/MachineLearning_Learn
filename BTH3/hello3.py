import pandas as pd
dulieu = pd.read_csv('house.csv', index_col=0)
dulieu.iloc[1:5,]

from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(iris_dt.data, iris_dt.target, test_size = 1/3.0, random_state=5)

X_train, X_test, y_train, y_test = train_test_split(dulieu.iloc[:,1:5], dulieu.iloc[:,0], test_size=1/3.0, random_state=100  )
X_train.iloc[1:5,]
X_test[1:5]
y_test[1:5]

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)
y_test[1:5]
y_pred[1:5]

from sklearn.metrics import mean_squared_error
err = mean_squared_error(y_test, y_pred)
err
import numpy as np
print(np.sqrt(err))