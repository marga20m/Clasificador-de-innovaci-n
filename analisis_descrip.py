import pandas as pd
x =pd.read_csv('tw')


#Proporción innovación en datos no balanceados
import matplotlib.pyplot as plt
i = sum(x['innov'] == 1)
ni = sum(x['innov'] == 0)
tweets = [i,ni]
nombres = ["Innovación","No Innovación"]
colores = ['#00FFFF','#4169E1']
plt.pie(tweets, labels=nombres, autopct="%0.1f %%", colors=colores)
plt.axis("equal")
plt.title('Porcentaje de tweets con innovación')
plt.show()



#Nube de palabras
tin = []
for n,i in enumerate(x['innov']):
    if i ==1:
        tin.append(x['text'][n])
        
import re
import string
import emoji
from nltk.corpus import stopwords
def preprocesamiento_nube(p):
    frases = ''
    for f in p:
        frases += f
    frases = frases.lower()
    frases = re.sub(r"http\S+|www\S+|https\S+",'',frases)
    frases = re.sub(r'\@\w+|\#','',frases)
    frases = frases.translate(str.maketrans('','',string.punctuation))
    frases = emoji.get_emoji_regexp().sub(r'', frases)
    return ' '.join([word for word in frases.split(' ') if word not in stopwords.words('spanish')])

from wordcloud import WordCloud
wordcloud = WordCloud(collocations = False, background_color = 'white').generate(preprocesamiento_nube(tin))
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")



#Innovación por cuentas
cuentas = ['JomaSport', 'panter_calzado', 'Callaghan_Shoes', 'Pikolinos', 'CalzadoPitillos','MAGRITshoes',
           'CalzadosRobusta','DianCalzado','FalSeguridad','pmcalzado','AlpeWomanShoes','DrCutillas',
           'MartinelliShoes','snipe_shoes','GARVALIN', 'NaturalWorldEco','DAngelaShoes','LuisGonzaloShoe',
          'CalzadosSegarra', 'calzadomiralles']

d = {}
for cuenta in cuentas:
    suma = 0
    for n,i in enumerate(x['cuenta']):
        if i == cuenta and x['innov'][n] == 1:
            suma += 1
    d.update({cuenta:suma})
from collections import OrderedDict
p = {}
for i in d.items():
    p.update({i[0] :(i[1] * 100)/50})
    
sortedDict = OrderedDict(sorted(p.items(), key=lambda x: x[1], reverse = True))
d = sortedDict
import matplotlib.pyplot as plt
keys = list(d.keys())
values = list(d.values())

plt.figure(figsize = (10, 5))

plt.bar(keys, values, color ='#4169E1',
		width = 0.9)

plt.xlabel("Cuentas de empresas")
plt.ylabel("Porcentaje de tweets con innovación")
plt.title("Porcentaje de tweets con innovación en cada cuenta")
plt.xticks(rotation = 90)
plt.show()