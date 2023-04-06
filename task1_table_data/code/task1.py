from sklearn.datasets import fetch_covtype

dataset = fetch_covtype().to_csv('iris.csv', encoding='utf-8')
x = dataset.data
y = dataset.target

print(x.shape)
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn.model_selection import cross_val_score
# from sklearn import datasets
# from sklearn import svm

# X, y = datasets.load_iris(return_X_y=True)
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# clf = svm.SVC(kernel='linear', C=1, random_state=42)
# scores = cross_val_score(clf, X, y, cv=5)
# print("%0.2f accuracy with a standard deviation of %0.2f" % (scores.mean(), scores.std()))
# print(X_train.shape, y_train.shape)
