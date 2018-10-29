# -*- coding: utf-8 -*-
#region import library
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import numpy as np
#endregion
#region import data
df = pd.read_csv("data.csv") # dataset kanserli hücrelerin verileri içeren bir datasettir. Aşağıda iyi huylu ve Kötü huylu diye iki klasa ayırıyoruz.
#endregion
#region data cleaning
df.drop(['Unnamed: 32','id'],axis = 1,inplace = True) # axis=1 bütün column'u drop eder. inplace = True datayı tekrar df içine kaydet anlamına geliyor.
# df.info()
df.diagnosis = [1 if each == 'M' else 0 for each in df.diagnosis] # modelin öğrenebilmesi için verileri 1 ve 0 lara dönüştürüyoruz
# print df # sınıflandırmada kullanabilmek için değerler yukarıda int dönüştürüldü.
#endregion
#region clasification and normalization
y = df.diagnosis.values
x_data = df.drop(['diagnosis'],axis = 1)
x = (x_data - np.min(x_data) / (np.max(x_data) - np.min(x_data))).values # bütün sutunların değerleri sıfırla bir arasında değerler haline sokularak data normalize edilir. Böylece uçlardaki değerlerin modeli bozması önlenir.
#endregion
#region train test splite
# random_state herzaman aynı şekilde işlem yapması için belirlenir, yoksa her seferinde farklı bölme işlemi yapar ve sürekli farklı sonuç buluruz.
x_train, x_test, y_train,  y_test = train_test_split(x, y, test_size = 0.2, random_state = 42) # Datayı train ve test olarak ikiye ayırdık. datanın 80% ile modelimizi eğiteceğiz, 20% ile test edeceğiz
x_train  = x_train.T # satır ve sütunları ters çevirme işlemi yapıyoruz. T: transpoze. Rowlar future oluyor.
x_test  = x_test.T
y_train  = y_train.T
y_test = y_test.T

# print "x_train: ", x_train.shape
# print "x_test : ", x_test.shape
# print "y_train: ", y_train.shape
# print "y_test : ", y_test.shape
#endregion
#region paremeter initialize and sigmoid funtion
def initialize_weight_and_bias(dimension):
    """ 
    Başlangıç weigths ve bias değerini belilemek için kullanılan foksiyon
    """
    w = np.full((dimension, 1), 0.01) # dimension değeri kaçar 0.01 lerden oluşan matrix verir. 0 verirsek model öğrenmez bu yüzden 0.01 veriyoruz.
    b = 0.0
    return w, b

# w, b = initialize_weight_and_bias(30)

def sigmoid_func(z):
    """  
    Sigmoid function f(z) = 1 / 1 + e^-z
    """
    y_head = 1 / (1 + np.exp(-z))
    return y_head

# print sigmoid_func(0)

def forward_backward_propagation(w,b,x_train,y_train):
    """
    loss = -(1 - y) * log(1 - y^) + y * log(y^)
    weight ve bios güncellmek için kullanılacak
    """
    # forward propagation
    z = np.dot(w.T, x_train) + b # iki matrix çarpımı için .dot foksiyonunu kullanıyoruz. değerler weightlerle çarpılıp bias ile toplanıyor.
    y_head = sigmoid_func(z) # z değeri sigmoid funtion'a sokulması gerkiyor.
    loss = -y_train * np.log(y_head) - (1 - y_train ) * np.log(1 - y_head) # kayıpları hesaplamak için loss kullanılır, sabit bir formül
    loss = loss[~np.isnan(loss)] # Bu satır nan değerleri filtrelemk için eklendi
    loss = [loss]
    cost = (np.sum(loss)) / x_train.shape[1]

    # backward propagation
    derivative_weight = (np.dot(x_train,((y_head - y_train).T))) / x_train.shape[1] # Bu formül deep learling kursunda öğrenilecek. cost'un weigth'e göre türevi
    derivative_bias = np.sum(y_head - y_train) / x_train.shape[1] # cost'un bias'a göre türevi alınır
    gradients = {"derivative_weight":derivative_weight, "derivative_bias":derivative_bias} # türevler dictionaryde depolanır.
    
    return cost, gradients

def update(w, b, x_train, y_train, learning_rate, number_of_iteration):
    cost_list = []
    cost_list2 = []
    index = []

    for i in range(number_of_iteration):
        cost, gradients = forward_backward_propagation(w, b, x_train, y_train)
        print "cost:", cost
        cost_list.append(cost)

        w = w - learning_rate * gradients["derivative_weight"]
        b = b - learning_rate * gradients["derivative_bias"]

        if i % 10 == 0:
            cost_list2.append(cost)
            index.append(i)
            print "Cost after iteration %i: %f" % (i, cost)

        paremeters = {"weight":w, "bias":b}
        plt.plot(index,cost_list2)
        plt.xticks(index, rotation = 'vertical')
        plt.xlabel('Number of iteration')
        plt.ylabel('Cost')
        plt.show()
        return paremeters, gradients, cost_list
#endregion
#region prediction
def predict(w, b, x_test):
    z = sigmoid_func(np.dot(w.T, x_test) + b)
    Y_predict = np.zeros((1,x_test.shape[1]))

    for i in range(z.shape[1]):
        if z[0,i] <= 0.5:
            Y_predict[0,i] = 0
        else:
            Y_predict[0,i] = 1
    
    return Y_predict
#endregion
#region logictic regression
def logistic_regression(x_train, y_train, x_test, y_test, learning_rate, num_iterations):
    dimension = x_train.shape[0] # 30
    w, b = initialize_weight_and_bias(dimension)
    paremeters, gradients, cost_list = update(w, b, x_train, y_train, learning_rate, num_iterations)
    y_predict_test = predict(paremeters['weight'], paremeters['weight'], x_test)

    print 'Test accuracy: {} %'.format(100 - np.mean(np.abs(y_predict_test - y_test)) * 100)

# logistic_regression(x_train, y_train, x_test, y_test, learning_rate = 3, num_iterations = 300)
#endregion
#region sklearn
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train.T,y_train.T) # T: transpose satırla sütünun yerini değiştirir.
print 'test accuracy: {}'.format(lr.score(x_test.T,y_test.T)) # predict edip bize accuracy'i döndüdrür.
#endregion