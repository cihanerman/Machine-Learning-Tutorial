# -*- coding: utf-8 -*-
#region import library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np
#endregion
#region import data
df = pd.read_csv("random_forest_regression_dataset.csv",header=None)
print df
x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)
#endregion
#region rendom foroest
rf = RandomForestRegressor(n_estimators=100, random_state=42) # random_state belirlenmesse kod her run edildiğinde farlı sonuçlanır.
rf.fit(x,y)
y_head = rf.predict(x)
#endregion
#region r_square
print 'r_score: ', r2_score(y,y_head)
print 'r_score: ',rf.score(x, y)
#endregion