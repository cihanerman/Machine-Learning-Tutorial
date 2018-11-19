# -*- coding: utf-8 -*-
#%%
#region import library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
#%%
#endregion
# region create dataset
# class1
x1 = np.random.normal(25,5,1000)
y1 = np.random.normal(25,5,1000)

# class2
x2 = np.random.normal(55,5,1000)
y2 = np.random.normal(60,5,1000)

# class3
x3 = np.random.normal(55,5,1000)
y3 = np.random.normal(15,5,1000)

x = np.concatenate((x1,x2,x3),axis=0)
y = np.concatenate((y1,y2,y3),axis=0)

dic = {'x':x,'y':y}

data = pd.DataFrame(dic)
# print data
data.info()
plt.scatter(x1,y1)
plt.scatter(x2,y2)
plt.scatter(x3,y3)
plt.show()
#endregion

# #region kmeans algoritması datayı aşağıdaki grafikteki gibi görecek
# data.info()
# plt.scatter(x1,y1,color='black')
# plt.scatter(x2,y2,color='black')
# plt.scatter(x3,y3,color='black')
# plt.show()
# #endregion

#region kmeans algo
wcss = []

for k in range(1,15):
    # en iyi wcss değerini bulma. Grafikte dirsek yapan kısım(tekrarların başladığı kısım)
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,15),wcss)
plt.xlabel('number of k (cluster) value')
plt.ylabel('wcss')
plt.show()
#endregion
#region k = 3 için model oluşturma
kmeans2 = KMeans(n_clusters=3)
clusters = kmeans2.fit_predict(data) # fit_predict: data yı fit et ve çıkan modeli dataya uygula ve clusterlarımı çıkar
data['label'] = clusters

plt.scatter(data.x[data.label == 0],data.y[data.label == 0], color='yellow')
plt.scatter(data.x[data.label == 1],data.y[data.label == 1], color='blue')
plt.scatter(data.x[data.label == 2],data.y[data.label == 2], color='green')
plt.scatter(kmeans2.cluster_centers_[:,0],kmeans2.cluster_centers_[:,1], color='red') # clusterların centroidleri
plt.show()
#endregion

