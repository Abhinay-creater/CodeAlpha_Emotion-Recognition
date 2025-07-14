import numpy as np
import librosa
import os
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Function to extract MFCC features
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=2.5, sr=22050)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return np.mean(mfccs.T, axis=0)

# Load data (Assuming files are organized in folders by emotion)
def load_dataset(data_dir):
    x = []
    y = []
    
    for emotion in os.listdir(data_dir):
        for file in os.listdir(os.path.join(data_dir, emotion)):
            file_path = os.path.join(data_dir, emotion, file)
            features = extract_features(file_path)
            x.append(features)
            y.append(emotion)  # Encode emotions numerically if needed
            
    return np.array(x), np.array(y)

# Define the model
def create_model(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(256, activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(tf.keras.layers.Dense(64, activation='softmax'))
    return model

# Main Execution
data_dir = 'path_to_your_dataset'
X, y = load_dataset(data_dir)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = create_model((X_train.shape[1],))
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50)
