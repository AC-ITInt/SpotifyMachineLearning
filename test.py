import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression

# Read the data
songs = pd.read_csv('spotify_playlist.csv')

# Assuming you have loaded and preprocessed your data
# Define features and target variable
X = songs[['Dance', 'Energy', 'Happy']]  # Features for both models
y = songs['Popularity']  # Target variable for both models

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the MLPRegressor model
mlp_regressor = MLPRegressor(hidden_layer_sizes=(3000,), max_iter=500, random_state=42)
mlp_regressor.fit(X_train, y_train)

# Create and train the Linear Regression model
linear_regressor = LinearRegression()
linear_regressor.fit(X_train, y_train)

# Get predictions for the test set
mlp_predictions = mlp_regressor.predict(X_test)
linear_predictions = linear_regressor.predict(X_test)

# Combine predictions from both models into a DataFrame
output_df = pd.DataFrame({
    'MLPRegressor_Prediction': mlp_predictions,
    'LinearRegression_Prediction': linear_predictions
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