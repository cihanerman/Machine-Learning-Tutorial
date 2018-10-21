# -*- coding: utf-8 -*-
#region import library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
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
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = rf.predict(x_)
#endregion
#region plot
plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("trubune level")
plt.ylabel("price")
plt.show()
#endregion