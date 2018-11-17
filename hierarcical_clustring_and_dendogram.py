# -*- coding: utf-8 -*-
#region import library
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
import numpy as np
#endregion
# region create dataset
# class1
x1 = np.random.normal(25,5,100)
y1 = np.random.normal(25,5,100)

# class2
x2 = np.random.normal(55,5,100)
y2 = np.random.normal(60,5,100)

# class3
x3 = np.random.normal(55,5,100)
y3 = np.random.normal(15,5,100)

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
#region dendegram
merge = linkage(data,method='ward') # ward data içerisindeki yayılımları minimase et
dendrogram(merge,leaf_rotation=90) # leaf_rotation noktalrın texttine 90 derece çeviriyor
plt.xlabel('dendogram')
plt.ylabel('eucliden distance')
plt.show()
#endregion
#region HC
hierarchy_cluster = AgglomerativeClustering(n_clusters=3,affinity='euclidean',linkage='ward') # n_clusters: dendogramdan bulunur. ward: cluster yayılımını en düşük seviyede tutmak için kullanılan method.
clusters = hierarchy_cluster.fit_predict(data) # fit_predict: modeli oluştur ve dataya göre cluster et.
data['label'] = clusters
plt.scatter(data.x[data.label == 0],data.y[data.label == 0], color='yellow')
plt.scatter(data.x[data.label == 1],data.y[data.label == 1], color='blue')
plt.scatter(data.x[data.label == 2],data.y[data.label == 2], color='green')
plt.show()
#endregion