# -*- coding: utf-8 -*-
#region import library
import pandas as pd
import numpy as np
import re
import nltk  as nlp # natural language tool kit
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer # bag of words yaratmak için kullanılan method
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
#endregion
#region import data
data = pd.read_csv(r'gender-classifier.csv', encoding='latin1') # r: okumak encoding='latin1': datanın latin harfleri içerdiğini bildiriyor
data = pd.concat([data.gender, data.description], axis= 1) # data clean
data.dropna(axis = 0, inplace = True) # dropna nan valuelerı drop ediyor
data.gender = [1 if each == 'female' else 0 for each in data.gender] # classification
# print data
#endregion
#region cleaning data
# regular expression
data_description = data.description[4]
print data_description
description = re.sub('[^a-zA-Z]', ' ', data_description) # ^ işareti bulma anlamına geliyor
description = description.lower()
print description
#endregion
#region stopwords (irrelavant words) gereksiz kelimeler öğr: the, and, as vb.
# nltk.download('stopwords')
# nltk.download('punkt')
# nlp.download('wordnet')
# description = description.split()
description = nlp.word_tokenize(description) # split yerine kullanıyoruz grmeri de algılayarak bölme işlemi yapıyor öğrn: isn't = > ['is','not']
print description
description = [word for word in description if not word in set(stopwords.words('english'))] # set unique olanları bul
print description
#endregion
#region lemmatization loved = > love, gitmek = > git. Kelimelerin köklerini bulmak
lemma = nlp.WordNetLemmatizer()
description = [lemma.lemmatize(word) for word in description]
print description
description = ' '.join(description)
print description
#endregion
#region All data cleaning
description_list = []
for description in data.description:
    description = re.sub('[^a-zA-Z]', ' ', description)
    description = description.lower()
    description = nlp.word_tokenize(description)
    # description = [word for word in description if not word in set(stopwords.words('english'))] # bu satır yavaşlatıyor bunu daha hızlı bir yöntemi aşağıda var
    lemma = nlp.WordNetLemmatizer()
    description = [lemma.lemmatize(word) for word in description] 
    description = ' '.join(description)
    description_list.append(description)

# print description_list
#endregion
#region bag of words
max_features = 500
count_vectorizer = CountVectorizer(max_features=max_features,stop_words='english') # lower işlemi lowercase paremetresi ile ve tokenize işlemi de oken_patter parametresi ile bu metodda yapılabiliyor
sparce_matrix = count_vectorizer.fit_transform(description_list).toarray() # fit_transform: fitet ve sparce_matrix' e uyarla, sparce_matrix = x
# print 'En sık kullanılan {} kelimeler: {}'.format(max_features,count_vectorizer.get_feature_names())
#endregion
#region text classification, model oluşturma
x = sparce_matrix
y = data.iloc[:,0].values # male or female classes
# tarin test splite
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.1, random_state = 42)
# naive bayes modeli 
gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_pre = gnb.predict(x_test)
print 'accuracy : ',gnb.score(y_pre.reshape(-1,1), y_test)
print 'accuracy : ',gnb.score(x_test, y_test)
#endregion