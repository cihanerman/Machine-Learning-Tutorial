# -*- coding: utf-8 -*-
#region import library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import numpy as np
#endregion
#region import data
df = pd.read_csv("data.csv") # dataset kanserli hücrelerin verileri içeren bir datasettir. Aşağıda iyi huylu ve Kötü huylu diye iki klasa ayırıyoruz.
#endregion
#region data cleaning
df.drop(['Unnamed: 32','id'],axis = 1,inplace = True) # axis=1 bütün column'u drop eder. inplace = True datayı tekrar df içine kaydet anlamına geliyor.
# df.info()
# print df # sınıflandırmada kullanabilmek için değerler yukarıda int dönüştürüldü.
#endregion
#region verileri ikiye ayırma
M = df[df.diagnosis == 'M']
B = df[df.diagnosis == 'B']
M.info()
B.info()
plt.scatter(M.radius_mean,M.texture_mean,color='red',label='Bad',alpha=0.3)
plt.scatter(B.radius_mean,B.texture_mean,color='green',label='Good',alpha=0.4)
plt.xlabel('radius_mean')
plt.ylabel('texture_mean')
plt.legend()
plt.show()
#endregion
#region normalization
df.diagnosis = [1 if each == 'M' else 0 for each in df.diagnosis] # modelin öğrenebilmesi için verileri 1 ve 0 lara dönüştürüyoruz
y = df.diagnosis
x_data = df.drop(['diagnosis'],axis=1)
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))
# print x
#endregion
#region train and test splite
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 1)
#endregion
#region svm
svm = SVC(random_state=1)
svm.fit(x_train, y_train)
print 'accuracy of svmalgo: ',svm.score(x_test, y_test)
#endregion