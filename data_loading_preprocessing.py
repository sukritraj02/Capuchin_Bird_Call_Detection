import numpy as np
import librosa

def load_audio_files(file_paths, sample_rate=22050):
    audio_data = []
    for path in file_paths:
        audio, sr = librosa.load(path, sr=sample_rate)
        audio_data.append(audio)
    return audio_data

def extract_mfcc_features(audio_data, sample_rate=22050, n_mfcc=13):
    mfcc_features = []
    for audio in audio_data:
        mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        mfcc_features.append(mfcc_scaled)
    return np.array(mfcc_features)
