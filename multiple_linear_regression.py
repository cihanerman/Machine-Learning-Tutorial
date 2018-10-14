# -*- coding: utf-8 -*-
#region import library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
#endregion
#region import data
df = pd.read_csv("multiple_linear_regression_dataset.csv")
print df
#endregion
#region MLR işlemi
x = df.iloc[:,[0,2]].values
y = df.maas.values.reshape(-1,1)

multiple_linear_reg = LinearRegression()
multiple_linear_reg.fit(x,y)
print 'b0: ', multiple_linear_reg.intercept_
print 'b1,b2: ',multiple_linear_reg.coef_
# deneyimler = np.array([x for x in df.deneyim.values]).reshape(1,-1)
# yaslar = np.array([x for x in df.yas.values]).reshape(1,-1)
# print deneyimler
# print yaslar
matrix = np.column_stack((deneyimler,yaslar))
print matrix
# predict
print multiple_linear_reg.predict(np.array([[10,35],[5,35]]))
#endregion