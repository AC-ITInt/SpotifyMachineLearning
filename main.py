#Spotify Machine Learning Project
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
#import matplotlib_inline.backend_inline

# Read the data
songs = pd.read_csv('spotify_playlist.csv')
songs.drop(
  [
    'Spotify Track Id',
    'Spotify Track Img',
    'Song Preview'
  ], 
  axis=1, inplace=True)
#plt.scatter(x=songs['Dance'], y=songs['BPM'], c=songs['Energy'], alpha = 0.5)
#plt.scatter(x=songs['Happy'], y=songs['Loud'], c=songs['Popularity'], alpha = 0.5)
plt.scatter(x=songs['Loud'],
            y=songs['Dance'],
            s=songs['Popularity'],
            c=songs['Happy'],
            alpha=.5)
#plt.show()

#pd.plotting.scatter_matrix( )

genres_split = songs['Parent Genres'].str.split(',', expand=True)

genres_split.columns = [f'Genre{i+1}' for i in range(genres_split.shape[1])]

songs2 = pd.concat([songs, genres_split], axis=1)

print(songs2[[
  'Genre1',
  'Genre2',
  'Genre3',
  'Genre4',
  'Genre5'
]])

print(songs2['Genre1'].unique())
songs2.to_csv('test.csv', index=False)
#TODO: Convert genres into numerical values