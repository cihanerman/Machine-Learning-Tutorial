# -*- coding: utf-8 -*-
#region import library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression  #Machine learning library
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
#endregion
#region import data
df = pd.read_csv("polynomial-regression.csv")
# print df
#endregion
#region ploted
y = df.araba_max_hiz.values.reshape(-1,1)
x = df.araba_fiyat.values.reshape(-1,1)
plt.scatter(x,y)
plt.xlabel('araba_fiyat')
plt.ylabel('araba_max_hiz')
# plt.show()
#endregion
#region regression
lr = LinearRegression()
lr.fit(x,y)
y_head = lr.predict(x)
# print x
# print y_head
plt.plot(x,y_head,color="red",label="linear")
# plt.show()
print lr.predict(10000)
#endregion
#region polynomial regression
pr = PolynomialFeatures(degree=4) #degree = n 
x_polynomial = pr.fit_transform(x) # fit_transform x'i kullanan ve ikinci dereceden PolynomialFeatures çevirir.
# print x_polynomial
lr2 = LinearRegression()
lr2.fit(x_polynomial,y)
y_head2 = lr2.predict(x_polynomial)
plt.plot(x,y_head2,color="black",label="polynomial")
plt.legend()
plt.show()
#endregion