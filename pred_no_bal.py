tok = tokenize2(x['text'])

from sklearn.model_selection import train_test_split
train_text, test_text, train_labels, test_labels = train_test_split(tok, x["innov"], stratify=x["innov"])
print(f"Training examples: {len(train_text)}, testing examples {len(test_text)}")

from sklearn.feature_extraction.text import CountVectorizer
real_vectorizer = CountVectorizer()
train_X = real_vectorizer.fit_transform(train_text)
test_X = real_vectorizer.transform(test_text)
voc = real_vectorizer.get_feature_names_out()


import numpy as np
def eliminar(train_X,voc):
    elim = []
    for n,v in enumerate(np.einsum('ij->j', train_X.toarray()) ==1): #palabras que solo aparecen 1 vez
        if v == True:
            elim.append(n)

    for n,v in enumerate(voc):
        if len(v) <=2: #palabras con menos de 2 caracteres
            elim.append(n)

    elim = np.unique(elim)        
    v = np.delete(voc,elim)
    return(v,elim)


v, elim = eliminar(train_X,voc)
train = np.delete(train_X.toarray(), elim, axis=1)
test = np.delete(test_X.toarray(), elim, axis=1)

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
def mostrar_resultados(y_test, pred_y):
    conf_matrix = confusion_matrix(y_test, pred_y)
    plt.figure(figsize=(8, 8))
    sns.heatmap(conf_matrix, annot=True,fmt="d");
    plt.title("Matriz de confusión")
    plt.ylabel('Clase real')
    plt.xlabel('Clase predicha')
    plt.show()
    print (classification_report(y_test, pred_y))
    

#Naive Bayes
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import roc_auc_score
mnb = MultinomialNB()
mnb.fit(train, train_labels)
y_pred = mnb.predict(test)
accuracy = accuracy_score(test_labels, y_pred)
print(f"Accuracy: {accuracy:.4%}")
auc = roc_auc_score(test_labels, y_pred)
print(f"Auc: {auc:.4%}")
mostrar_resultados(test_labels, y_pred)





#Soporte Vectores
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
svc = LinearSVC(C= 1, class_weight = 'balanced') #Indicamos que las clases están desbalanceadas
svc.fit(train, train_labels)
pred = svc.predict(test)
accuracy = accuracy_score(test_labels, pred)
print(f"Accuracy: {accuracy:.4%}")
auc = roc_auc_score(test_labels, pred)
print(f"Auc: {auc:.4%}")
mostrar_resultados(test_labels, pred)
#palabras más influyentes
n = svc.fit(train, train_labels)
dic = {}
for r,i in enumerate(v):
    dic[i] = n.coef_[0][r]
top_cinco = []

for r in range(5):  
    maximo = max(dic, key = dic.get)  
    top_cinco.append(maximo)
    del dic[maximo]  
print(top_cinco)
dic = {}
for r,i in enumerate(v):
    dic[i] = n.coef_[0][r]

p = {}
for i in top_cinco:
    p[i] = dic[i]
    

keys = list(p.keys())
values = list(p.values())
fig = plt.figure(figsize = (10, 5))
plt.bar(keys, values, color ='#191970',
		width = 0.9)

plt.xlabel("")
plt.ylabel("Valor del coeficiente")
plt.title("Palabras más influyentes para clasificar como innovación")
plt.xticks(rotation = 45)
plt.show()    
    
    


#Regresión Logística
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=20, class_weight = 'balanced', n_jobs = 4) #Indicamos que las clases están desbalanceadas
lr.fit(train, train_labels)
pred2 = lr.predict(test)
accuracy2 = accuracy_score(test_labels, pred2)
print(f"Accuracy: {accuracy2:.4%}")
lr_auc = roc_auc_score(test_labels, pred2)
print(f"Auc: {lr_auc:.4%}")
mostrar_resultados(test_labels, pred2)
#palabras más influyentes
n = lr.fit(train, train_labels)
dic = {}
for r,i in enumerate(v):
    dic[i] = n.coef_[0][r]
top_cinco = []

for r in range(5):  
    maximo = max(dic, key = dic.get)  
    top_cinco.append(maximo)
    del dic[maximo]  
print(top_cinco)
dic = {}
for r,i in enumerate(v):
    dic[i] = n.coef_[0][r]

p = {}
for i in top_cinco:
    p[i] = dic[i]
    

keys = list(p.keys())
values = list(p.values())
fig = plt.figure(figsize = (10, 5))
plt.bar(keys, values, color ='#8A2BE2',
		width = 0.9)

plt.xlabel("")
plt.ylabel("Valor del coeficiente")
plt.title("Palabras más influyentes para clasificar como innovación")
plt.xticks(rotation = 45)
plt.show()    
    



#CURVA ROC
from sklearn.metrics import roc_curve
def plot_roc_curve(fper, tper, fper1, tper1, fper2, tper2):
    plt.figure(figsize = (10, 5))
    plt.plot(fper, tper, color='red', label='Regresión Logística')
    plt.plot(fper1,tper1, color = 'blue', label = 'Soporte Vectores')
    plt.plot(fper2,tper2, color = 'yellow', label = 'Naive Bayes')
    plt.plot([0, 1], [0, 1], color='green', linestyle='--')
    plt.xlabel('Ratio Falsos Positivos')
    plt.ylabel('Ratio Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend()
    plt.show()
    
fper, tper, thresholds = roc_curve(test_labels, pred2)
fper1, tper1, thresholds1 = roc_curve(test_labels, pred)
fper2, tper2, thresholds2 = roc_curve(test_labels, y_pred)

plot_roc_curve(fper, tper,fper1, tper1,fper2, tper2)

lr_auc = roc_auc_score(test_labels, pred2)
print(f"Auc Regresión Logística: {lr_auc:.4%}")
svc_auc = roc_auc_score(test_labels, pred)
print(f"Auc Soporte Vectores: {svc_auc:.4%}")
nb_auc = roc_auc_score(test_labels, y_pred)
print(f"Auc Naive Bayes: {nb_auc:.4%}")