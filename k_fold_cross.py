# -*- coding: utf-8 -*-
#%% import library
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
#%% import data
iris = load_iris()
# print(iris)
x = iris.data
y = iris.target
#%% normalization
x = (x - np.min(x)) / (np.max(x)  - np.min(x))
#%% train test splite
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)
#%% knn model
knn = KNeighborsClassifier(n_neighbors=3) # k:n_neighbors
#%% knn fold cv k =10
accuracies = cross_val_score(estimator = knn, X = x_train, y=y_train, cv= 10) # estimator: Machine learnin algorştması cv: k değeri
print(accuracies)
print('average accuracies: ', np.mean(accuracies))
print('average std: ', np.std(accuracies)) # yayılım
#%% knn model test
knn.fit(x_train, y_train)
print('test accuracy: ',knn.score(x_test, y_test))


#%%
