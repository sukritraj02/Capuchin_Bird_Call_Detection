def predict_bird_call(model, new_audio_mfcc):
    prediction = model.predict(np.array([new_audio_mfcc]))  # Input must be 2D array
    return int(prediction[0] > 0.5)  # Binary classification: 1 for bird call, 0 otherwise
