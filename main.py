from data_loading_preprocessing import load_audio_files, extract_mfcc_features
from data_visualization import visualize_audio_data, visualize_mfcc
from tensorflow_dataset import create_tf_dataset
from train_test_split import train_test_split_data
from deep_learning_model import build_model
from prediction_classification import predict_bird_call
from forest_recording_analysis import analyze_forest_recordings
from postprocessing_export import export_results_to_csv, plot_confusion_matrix

# Example flow
file_paths = [
    r'C:\Users\sukri\OneDrive\Desktop\DeepAudioClassifier\data\Parsed_Capuchinbird_Clips\XC3776-0.wav',
    r'C:\Users\sukri\OneDrive\Desktop\DeepAudioClassifier\data\Parsed_Not_Capuchinbird_Clips\afternoon-birds-song-in-forest-1.wav'
]
audio_data = load_audio_files(file_paths)
mfcc_features = extract_mfcc_features(audio_data)

# Verify the shape of mfcc_features
print(f'MFCC features shape: {mfcc_features.shape}')  # Expecting (num_samples, n_mfcc, time_steps)

# Visualize audio data and MFCC features
visualize_audio_data(audio_data[0])
if mfcc_features[0].ndim == 2:  # Ensure it's 2D
    visualize_mfcc(mfcc_features[0])
else:
    print("MFCC features for the first audio sample is not 2D!")

# Splitting data and training
labels = [1, 0]  # Example labels
X_train, X_test, y_train, y_test = train_test_split_data(mfcc_features, labels)
model = build_model(X_train.shape[1])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Predictions and evaluation
forest_predictions = analyze_forest_recordings(model, X_test)
export_results_to_csv(forest_predictions)
plot_confusion_matrix(y_test, model.predict(X_test).round())
