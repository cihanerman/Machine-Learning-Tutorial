# -*- coding: utf-8 -*-
#region import library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
#endregion
#region import data
df = pd.read_csv("linear_regression_dataset.csv")
print df
#endregion
#region data plot
plt.scatter(df.deneyim,df.maas)
plt.xlabel('deneyim')
plt.ylabel('maas')
plt.show()
#endregion
#region sklearn
linear_reg = LinearRegression()
x = df.deneyim.values.reshape(-1,1) # sklearn kütüpanesi için shape düzeltilmeli öğr:(14,) (14,1) şekline dönüştürülmeli bunun için reshape metodu kullanılır
y = df.maas.values.reshape(-1,1) # values np array'e dönüştürmek için
print type(x)
print type(y)
# fit etmek
linear_reg.fit(x,y)
print linear_reg
#prediction
b0 = linear_reg.predict(0) # b0 bulma 1. yol (intersept: y eksenini kestiği nokta)
b0_ = linear_reg.intercept_ # b0 bulma 2. yol
print 'b0: ',b0
print 'b0_: ',b0_
b1 = linear_reg.coef_ # b1(eğim,slope) bulma
print 'b1: ',b1
maas_yeni = b0 + b1 * 11 # 11 deneyim
print 'mass yeni: ',maas_yeni
print 'maas yeni2: ', linear_reg.predict(11)
#endregion
#region visualize line
arrray = np.array([x for x in range(0,16)]).reshape(-1,1)
print arrray.shape
plt.scatter(df.deneyim,df.maas)
y_heads = linear_reg.predict(arrray)
plt.plot(arrray,y_heads,color = 'red')
plt.show()
#endregion