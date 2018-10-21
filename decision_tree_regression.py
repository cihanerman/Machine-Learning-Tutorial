# -*- coding: utf-8 -*-
#region import library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import numpy as np
#endregion
#region import data
df = pd.read_csv("decision_tree_regression_dataset.csv",header=None)
print df
x = df.iloc[:,0].values.reshape(-1,1)
y = df.iloc[:,1].values.reshape(-1,1)
#endregion
#region decision tree regression
tree_reg = DecisionTreeRegressor()
tree_reg.fit(x,y)
tree_reg.predict(5.5)
x_ = np.arange(min(x),max(x),0.01).reshape(-1,1)
y_head = tree_reg.predict(x_)
#endregion
#region plot
plt.scatter(x,y,color="red")
plt.plot(x_,y_head,color="green")
plt.xlabel("tribune level")
plt.ylabel("price")
plt.show()
#endregion