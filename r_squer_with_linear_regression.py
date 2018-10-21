# -*- coding: utf-8 -*-
#region import library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import numpy as np
#endregion
#region import data
df = pd.read_csv("linear_regression_dataset.csv")
linear_reg = LinearRegression()
x = df.deneyim.values.reshape(-1,1) # sklearn kütüpanesi için shape düzeltilmeli öğr:(14,) (14,1) şekline dönüştürülmeli bunun için reshape metodu kullanılır
y = df.maas.values.reshape(-1,1) # values np array'e dönüştürmek için
#endregion
#region r_square
linear_reg.fit(x,y)
y_head = linear_reg.predict(x)
print 'r_square: ', r2_score(y,y_head)
#endregion