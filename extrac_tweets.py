bearer_token = ''

import requests
import tweepy

def get_data(url):
    headers = {'Authorization': "Bearer " + bearer_token }
    response = requests.get(url, headers=headers)
    response_data = response.json()
    return response_data

cuentas = ['JomaSport', 'Panter_Calzado', 'Callaghan_Shoes', 'pikolinos', 'CalzadoPitillos','MAGRITshoes',
           'CalzadosRobusta','DianCalzado','FalSeguridad','pmcalzado','AlpeWomanShoes','DrCutillas',
           'MartinelliShoes','snipe_shoes','GARVALIN', 'NaturalWorldEco','DAngelaShoes','LuisGonzaloShoe',
          'CalzadosSegarra', 'calzadomiralles']

get_data("https://api.twitter.com/2/users/by/username/calzadomiralles?expansions=pinned_tweet_id&user.fields=created_at&tweet.fields=created_at")
client = tweepy.Client(bearer_token=bearer_token)
tweets = client.get_users_tweets(id= '67268648', max_results= 15)


id = []
text = []
innov = []
for tweet in tweets.data:
    id.append(tweet.id)
    text.append(tweet.text)
    innov.append('')

import pandas as pd
dict = {'id': id, 'text': text, 'innov':innov} 
df = pd.DataFrame(dict) 
df.to_csv('tweets.csv')