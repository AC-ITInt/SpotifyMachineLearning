#Spotify Machine Learning Project
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MultiLabelBinarizer
#import matplotlib_inline.backend_inline

# Read the data
#thinking of using Dance, Energy, and Happy
songs = pd.read_csv('spotify_playlist.csv')
songs.drop(
  [
    'Artist',
    'BPM',
    'Album Date',
    'Time',
    'Acoustic',
    'Instrumental',
    'Speech',
    'Live',
    'Loud',
    'Key',
    'Time Signature',
    'Added At',
    'Spotify Track Id',
    'Album Label',
    'Camelot',
    'Spotify Track Img',
    'Song Preview'
  ], 
  axis=1, inplace=True)

# Split genres into separate columns and then create a binary representation
genres_split = songs['Parent Genres'].str.get_dummies(sep=', ')

# Rename columns to reflect genre names
genres_split.columns = [f'Genre_{name}' for name in genres_split.columns]
songs2 = pd.concat([songs, genres_split], axis=1)

def MLPCalc(InputGenre):
    
    # Prepare the data for MLP
    MLPX = songs2[genres_split.columns]  # Features are the genre columns
    MLPX = pd.DataFrame(MLPX, columns=MLPX.columns)
    MLPY = songs2['Popularity']  # Target variable
    
    # Split the dataset into training set and test set
    MLPX_train, MLPX_test, MLPY_train, MLPY_test = train_test_split(
      MLPX, MLPY, test_size=0.3, random_state=42
    )  # 70% training and 30% testing
    
    # Create and train the MLP model
    mlp = MLPRegressor(hidden_layer_sizes=(500), max_iter=3000, random_state=42)
    mlp.fit(MLPX_train, MLPY_train)
    
    # Predict the Popularity on the test set
    #MLPY_pred = mlp.predict(MLPX_test)
    
    genre = string_to_binary_df(InputGenre, MLPX.columns)
#'Hip Hop' => [[0,0,1,0,0,0,0,0,0,0,0,0,0]]

    return mlp.predict(genre)
    print("Predicted MLPPopularity:", mlp.predict(genre))

## Translate a string to a binary dataframe
def string_to_binary_df(input_string, genre_columns):
    binary_list = [0] * len(genre_columns)
    genreList = input_string.split(', ')
    for index, genre in enumerate(genre_columns):
        for genre_name in genreList:
            if genre_name.lower() in genre.lower():
                binary_list[index] = 1
    print(binary_list)
    return pd.DataFrame([binary_list], columns=genre_columns)

# Example usage

#print("Predicted MLPPopularity:", MLPOutput)

# Calculate the Mean Squared Error to evaluate performance
###mse = mean_squared_error(MLPY_test, MLPY_pred)
###print("Mean Squared Error:", mse)

#print(songs2['Parent Genres'].unique())
#songs2.to_csv('test.csv', index=False)

#plt.scatter(x=songs['Dance'], y=songs['BPM'], c=songs['Energy'], alpha = 0.5)
#plt.scatter(x=songs['Happy'], y=songs['Loud'], c=songs['Popularity'], alpha = 0.5)

#plt.scatter(x=songs['Happy'],
#            y=songs['Dance'],
#            s=songs['Popularity'],
#            c=songs['Energy'],
#            alpha=.5)
#plt.show()

#pd.plotting.scatter_matrix( )

def LRCalc(Dance, Energy, Happy):
    #Linear regression for Dance, Energy and Happy
    LRX = songs[['Dance', 'Energy', 'Happy']]
    LRX = pd.DataFrame(LRX, columns=LRX.columns)
    LRY = songs['Popularity']
    print('split')
    
    # Split the dataset into training and testing sets
    LRX_train, LRX_test, LRY_train, LRY_test = train_test_split(
      LRX[['Dance', 'Energy', 'Happy']], 
      LRY, test_size=0.3, random_state=42)
    
    # Create and train the linear regression model
    lr_model = LinearRegression()
    lr_model.fit(LRX_train, LRY_train)
    
    # Predict Popularity on the test set
    #LRY_pred = lr_model.predict(LRX_test)
    
    # Evaluate the model
    ###mse = mean_squared_error(LRY_test, LRY_pred)
    ###print("Mean Squared Error:", mse)
    
    # Make predictions on new data (example)
    new_song_features = [[Dance, Energy, Happy]]  # Dance, Energy, Happy
    return lr_model.predict(
      pd.DataFrame(new_song_features,columns=LRX.columns))
    
    print("Predicted LRPopularity:", LROutput)


# combined_predictions = pd.concat([MLPY_pred, LRY_pred], axis=1)

def SongClassifier(Dance, Energy, Happy, InputGenre):
    #creating dataframe from outputs
    output_df = pd.DataFrame({
        'MLPRegressor_Prediction': MLPCalc(InputGenre),
        'LinearRegression_Prediction': LRCalc(Dance, Energy, Happy)
    })
    # Define threshold for classification
    classification_threshold = 70  # Adjust as needed
    
    # Iterate through each row in the DataFrame
    for index, row in output_df.iterrows():
        # Combine predictions from MLPRegressor and Linear Regression
        combined_prediction = (row['MLPRegressor_Prediction'] + row['LinearRegression_Prediction']) / 2
    
        # Classify the song as "good" or "bad" based on the combined prediction and threshold
        classification = 'good' if combined_prediction >= classification_threshold else 'bad'
    
        # Store the classification result
        output_df.at[index, 'Classification'] = classification
    
    # Display the updated DataFrame with classification
    print(output_df)
    
    print("The song is classified as:", output_df.iloc[0]['Classification'])


def main():
    Dance = 100
    Energy = 70
    Happy = 80
    Genre = 'Hip Hop, Rock'
    SongClassifier(Dance, Energy, Happy, Genre)
main()