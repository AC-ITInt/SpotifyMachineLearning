#Spotify Machine Learning Project

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
#import matplotlib_inline.backend_inline

# Read the data
songs = pd.read_csv('spotify_playlist.csv', sep=',')