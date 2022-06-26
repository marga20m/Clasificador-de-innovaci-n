import time
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
def split(n):
    train_text, test_text, train_labels, test_labels = train_test_split(tokenize2(x["text"]), x["innov"], train_size = n, stratify=x["innov"])
    return(train_text, test_text, train_labels, test_labels)

from sklearn.feature_extraction.text import CountVectorizer
def vec_caract(train_text,test_text):
    real_vectorizer = CountVectorizer()
    train_X = real_vectorizer.fit_transform(train_text)
    test_X = real_vectorizer.transform(test_text)
    voc = real_vectorizer.get_feature_names_out()
    return(train_X,test_X,voc)

#Naive Bayes
from sklearn.naive_bayes import MultinomialNB
mnb = MultinomialNB()
sizes = [2,100,200,300,400,500,600,700,998]
times = []
for n in sizes:
    train_text, test_text, train_labels,test_labels = split(n)
    train_X,test_X,voc = vec_caract(train_text,test_text)
    start = time.perf_counter()
    mnb.fit(train_X, train_labels)
    end = time.perf_counter()
    times.append(end - start)
plt.plot(sizes,times)
plt.xlabel('Tamaño conjunto entrenanmiento')
plt.ylabel('Tiempo')
plt.title('Coste algorítmico Naive Bayes')
plt.show()



#Soporte Vectores
from sklearn.svm import LinearSVC
svc = LinearSVC(C= 1, class_weight = 'balanced')
sizes = [10,100,200,300,400,500,600,700,998]
times = []
for n in sizes:
    train_text, test_text, train_labels,test_labels = split(n)
    train_X,test_X,voc = vec_caract(train_text,test_text)
    start = time.perf_counter()
    svc.fit(train_X, train_labels)
    end = time.perf_counter()
    times.append(end - start)
plt.plot(sizes,times)
plt.xlabel('Tamaño conjunto entrenanmiento')
plt.ylabel('Tiempo')
plt.title('Coste algorítmico Soporte Vectores')
plt.show()



#Regresión Logística
from sklearn.linear_model import LogisticRegression
lr =  LogisticRegression(C=20, class_weight = 'balanced', n_jobs = 4)
sizes = [10,100,200,300,400,500,600,700,998]
times = []
for n in sizes:
    train_text, test_text, train_labels,test_labels = split(n)
    train_X,test_X,voc = vec_caract(train_text,test_text)
    start = time.perf_counter()
    lr.fit(train_X, train_labels)
    end = time.perf_counter()
    times.append(end - start)
plt.plot(sizes,times)
plt.xlabel('Tamaño conjunto entrenanmiento')
plt.ylabel('Tiempo')
plt.title('Coste algorítmico Regresión Logística')
plt.show()