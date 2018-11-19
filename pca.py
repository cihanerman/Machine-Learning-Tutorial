# -*- coding: utf-8 -*-
# %%
#region import library
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#endregion
#region import data
irish = load_iris()
data = irish.data
feature_names = irish.feature_names
y = irish.target
df = pd.DataFrame(data, columns = feature_names)
df['class'] = y
x = data
# print df
#endregion
#region PCA
pca = PCA(n_components= 2, whiten= True) # n_components: düşürülmek istenen boyutsayısı. whiten= True: normalize et anlamı taşıyor
pca.fit(x) # bu örnek için 4 boyutlu datayı 2 boyuta düşür analamı taşıyor
x_pca = pca.transform(x) # boyut düşürme işlemi burada gerçekleşiyor
print ('variance ratio: ',pca.explained_variance_ratio_)
print ('sum: ',sum(pca.explained_variance_ratio_)) # varyans
#endregion
#region Virtualize
df['p1'] = x_pca[:,0]
df['p2'] = x_pca[:,1]
# print df
colors = ['red','green','blue']

for each in range(3):
    plt.scatter(df.p1[df['class'] == each],df.p2[df['class'] == each],color = colors[each], label = irish.target_names[each])

plt.legend()
plt.xlabel('p1')
plt.ylabel('p2')
plt.show()
#endregion