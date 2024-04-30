import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE

import warnings
warnings.filterwarnings('ignore')

tracks = pd.read_csv('tracks.csv')
tracks.head()

tracks.shape
tracks.info()
tracks.isnull().sum()

tracks.dropna(inplace = True)
tracks.isnull().sum().plot.bar()
plt.show()

tracks = tracks.drop(['track_uri', 'album_id', 'track_href'], axis = 1)
tracks['track'].nunique(), tracks.shape

tracks = tracks.sort_values(by=['popularity'], ascending=False)
tracks.drop_duplicates(subset=['track'], keep='first', inplace=True)

plt.figure(figsize = (10, 5))
sb.countplot(tracks['artist'])
plt.axis('off')
plt.show()

floats = []
for col in tracks.columns:
    if tracks[col].dtype == 'float':
    	floats.append(col)
len(floats)

plt.subplots(figsize = (15, 5))
for i, col in enumerate(floats):
    plt.subplot(2, 5, i + 1)
    sb.distplot(tracks[col])
plt.tight_layout()
plt.show()

%%capture
song_vectorizer = CountVectorizer()
song_vectorizer.fit(tracks['type'])

tracks = tracks.sort_values(by=['popularity'], ascending=False).head(2000)

def get_similarities(song_name, data):
       
  # Getting vector for the input song.
    text_array1 = song_vectorizer.transform(data[data['track']==song_name]['type']).toarray()
    num_array1 = data[data['track']==song_name].select_dtypes(include=np.number).to_numpy()
   
  # We will store similarity for each row of the dataset.
    sim = []
    for idx, row in data.iterrows():
        name = row['track']
     
    # Getting vector for current song.
        text_array2 = song_vectorizer.transform(data[data['track']==name]['type']).toarray()
        num_array2 = data[data['track']==name].select_dtypes(include=np.number).to_numpy()
 
    # Calculating similarities for text as well as numeric features
        text_sim = cosine_similarity(text_array1, text_array2)[0][0]
        num_sim = cosine_similarity(num_array1, num_array2)[0][0]
        sim.append(text_sim + num_sim)
     
    return sim

def recommend_songs(song_name, data=tracks):
      # Base case
    if tracks[tracks['track'] == song_name].shape[0] == 0:
        print('This song is either not so popular or you\
                have entered invalid_name.\n Some songs you may like:\n')
     
        for song in data.sample(n=5)['track'].values:
            print(song)
        return
   
    data['similarity_factor'] = get_similarities(song_name, data)
 
    data.sort_values(by=['similarity_factor', 'popularity'],
                   ascending = [False, False],
                   inplace=True)
   
  # First song will be the input song itself as the similarity will be highest.
    display(data[['track', 'artist']][2:7])
  
recommend_songs('Shape of You')
recommend_songs('Rebellion')

