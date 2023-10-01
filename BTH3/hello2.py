from sklearn.datasets import load_iris
iris_dt = load_iris()
iris_dt.data[1:5]
iris_dt.target[1:5]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(iris_dt.data, iris_dt.target, test_size = 1/3.0, random_state=5)

from sklearn.model_selection import KFold 
kf = KFold (n_splits=15)
X = iris_dt.data
y = iris_dt.target
for train_index, test_index in kf.split(X):
    print("Train: ", train_index, "Test: ", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    print("X_test \n", X_test)
    print("==================")


    from sklearn.tree import DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth=3, min_samples_leaf=5)
clf_gini.fit(X_train, y_train)

y_pred = clf_gini.predict(X_test)
y_test
clf_gini.predict([[4,4,3,3]])


from sklearn.metrics import accuracy_score
print(" Accuracy is ", accuracy_score(y_test, y_pred)*100)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred, labels=[2,0,1])
print("array ", confusion_matrix(y_test, y_pred, labels=[2,0,1]))