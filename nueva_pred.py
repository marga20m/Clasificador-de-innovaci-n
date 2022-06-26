#Primero se entrena el modelo
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
train_text, test_text, train_labels, test_labels = train_test_split(tok, x["innov"], stratify=x["innov"])
real_vectorizer = CountVectorizer()
train_X = real_vectorizer.fit_transform(train_text)
test_X = real_vectorizer.transform(test_text)
voc = real_vectorizer.get_feature_names_out()


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

#Balanceado de los datos
from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(train, train_labels)
from sklearn.naive_bayes import MultinomialNB
mnbb = MultinomialNB()
mnbb.fit(X_res, y_res)


#Predicción naive bayes balanceado
texts = ['RT @YorokobuMag: De lejos parece asqueroso, pero de cerca es 😍😍😍 https://t.co/AXPOx6PBOT',
         'La nueva colección está repleta de modelos versátiles y atemporales, ¿La has visto ya? https://t.co/pptY8IsRnl #ecofriendly #vegan #shoesaddict',
         '😍😍😍 https://t.co/AVeWZCSV1K',
         'La belleza de la fabricación artesanal https://t.co/pptY8IsRnl #artesania https://t.co/hdH7XB6nqp',
         'La primera comunión es un acontecimiento único que los peques recordarán toda su vida, ¿Sabes que tienes que tener en cuenta a la hora de elegirlo? ¡Te lo contamos! https://t.co/BAvUkaDBvr #primeracomunión https://t.co/cp3G8oEUV3']


real_vectorizer = CountVectorizer(vocabulary = v)
frases_X = real_vectorizer.transform(tokenize2(texts))
predicciones = mnbb.predict(frases_X.toarray())
for text,innov in zip(texts,predicciones):
    print(f"{innov:5}-{text}")

