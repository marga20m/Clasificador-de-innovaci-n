import pandas as pd
import string
import re
import emoji
from nltk.corpus import stopwords

punctuation = set(string.punctuation) | {'¿','¡'}

def normalize(s):
    replacements = (
        ("á", "a"),
        ("é", "e"),
        ("í", "i"),
        ("ó", "o"),
        ("ú", "u"),
    )
    for a, b in replacements:
        s = s.replace(a, b).replace(a.upper(), b.upper())
    return s

def tokenize(sentence):
    tokens = []
    sentence = sentence.lower()
    sentence = re.sub(r"http\S+|www\S+|https\S+",'',sentence)
    sentence = re.sub(r'\@\w+|\#','',sentence)
    sentence = emoji.get_emoji_regexp().sub(r'', sentence)
    sentence = re.sub(r'[^a-zA-Zñ\s]', '', normalize(sentence))
    for token in sentence.split():
        new_token = []
        for character in token:
            if character not in punctuation:
                new_token.append(character)
        #if new_token and token not in stopwords.words('spanish'):
        if new_token and token not in [normalize(i) for i in stopwords.words('spanish')]:
            tokens.append("".join(new_token))
    return tokens

def tokenize2(texto):
    l = []
    for j in range(len(texto)):
        s = ''
        for i in tokenize(texto[j]):
            s = s+' '+i
        l.append(s)
    return l

x =pd.read_csv('tw')
tok = tokenize2(x['text'])