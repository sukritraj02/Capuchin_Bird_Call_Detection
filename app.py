import librosa
import numpy as np
import os
import tensorflow as tf
import pandas as pd
import gradio as gr
from sklearn.model_selection import train_test_split  
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Path to your audio data
DATA_PATH = r'C:\Users\sukri\OneDrive\Desktop\DeepAudioClassifier\data'

# Load and preprocess audio files
def load_audio_files(path, duration=3, sample_rate=16000):
    audio_data = []
    labels = []
    
    # Define the folders for Capuchin and Non-Capuchin bird calls
    folder_mapping = {
        'parsed_capuchinbird_clips': 1,
        'parsed_not_capuchinbird_clips': 0
    }
    
    for folder, label in folder_mapping.items():
        folder_path = os.path.join(path, folder)
        
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            # Ensure the file is a .wav file
            if file_path.endswith('.wav'):
                audio, _ = librosa.load(file_path, sr=sample_rate, duration=duration, mono=True)  # Load as mono
                audio = librosa.util.fix_length(audio, size=duration * sample_rate)  # Ensure 3 sec clips
                audio_data.append(audio)
                labels.append(label)
    
    return np.array(audio_data), np.array(labels)

# Extract MFCC features from audio with delta and delta-delta coefficients
def extract_features(audio, sample_rate=16000, n_mfcc=13):
    # Ensure the audio input is long enough for MFCC
    if len(audio) < sample_rate * 3:  # Less than 3 seconds
        audio = np.pad(audio, (0, sample_rate * 3 - len(audio)), mode='constant')
    
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Stack MFCC, delta, and delta-delta coefficients to form a larger feature set
    features = np.vstack([mfcc, mfcc_delta, mfcc_delta2])
    return np.mean(features, axis=1)  # Take mean across time frames for simplicity

# Load audio data
X, y = load_audio_files(DATA_PATH)

# Extract MFCC features
X_features = np.array([extract_features(audio) for audio in X])

# Check the shape of X_features
print("X_features shape:", X_features.shape)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

# Check for class imbalance
unique, counts = np.unique(y_train, return_counts=True)
print(f"Class distribution in training data: {dict(zip(unique, counts))}")

# Apply class weights to handle imbalance
from sklearn.utils import class_weight
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print(f"Class weights: {class_weight_dict}")

# Build the model (increased complexity)
model = Sequential([
    Dense(256, activation='relu', input_shape=(X_train.shape[1],)),  # Input shape must match features
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')  # Binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model with class weights
history = model.fit(X_train, y_train, epochs=30, batch_size=32, validation_data=(X_test, y_test), class_weight=class_weight_dict)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# Function to predict bird calls using the trained model
def predict_bird_call(audio):
    try:
        # Check if the audio input is a file path or the raw audio data
        if isinstance(audio, str):  # File path
            audio_data, sample_rate = librosa.load(audio, sr=16000, duration=3, mono=True)  # Load as mono
        elif isinstance(audio, tuple):  # Gradio passes a tuple containing (sample_rate, audio_array)
            sample_rate, audio_data = audio
            audio_data = audio_data.astype(np.float32)  # Ensure audio is float
            audio_data = librosa.to_mono(audio_data)  # Convert to mono
        else:
            return "Invalid audio input format."
        
        # Ensure the audio is 3 seconds long
        if len(audio_data) > sample_rate * 3:
            audio_data = audio_data[:sample_rate * 3]
        elif len(audio_data) < sample_rate * 3:
            audio_data = np.pad(audio_data, (0, sample_rate * 3 - len(audio_data)), mode='constant')
        
        # Extract features
        features = extract_features(audio_data, sample_rate=sample_rate)
        
        # Check for invalid features
        if np.isnan(features).any() or np.isinf(features).any():
            return "Invalid audio input detected."

        # Ensure features have the correct shape
        features = features.reshape(1, -1)  # Reshape to (1, n_features)
        
        # Make prediction
        prediction = model.predict(features)[0][0]
        return "Capuchin Bird Call Detected" if prediction > 0.5 else "No Capuchin Bird Call Detected"

    except Exception as e:
        return f"Error processing audio: {str(e)}"

# Deploy the model using Gradio for an interactive interface
interface = gr.Interface(fn=predict_bird_call, inputs="audio", outputs="text")
interface.launch(share=True)

# Save predictions to CSV
predictions = model.predict(X_test)
df = pd.DataFrame({'Actual': y_test, 'Predicted': predictions.flatten()})
df['Predicted_Label'] = ['Capuchin Bird Call Detected' if pred > 0.5 else 'No Capuchin Bird Call Detected' for pred in predictions]
df.to_csv('Capuchinbird_Predictions.csv', index=False)

# Compute confusion matrix
y_pred = (predictions.flatten() > 0.5).astype(int)  # Convert probabilities to binary predictions
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Capuchin', 'Capuchin'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.show()
