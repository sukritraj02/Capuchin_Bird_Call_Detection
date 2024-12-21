import matplotlib.pyplot as plt
import librosa.display

def visualize_audio_data(audio, sr=22050):
    plt.figure(figsize=(10, 4))
    librosa.display.waveshow(audio, sr=sr)
    plt.title("Waveform")
    plt.show()

def visualize_mfcc(mfcc, sr=22050):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, x_axis='time', sr=sr)
    plt.colorbar()
    plt.title('MFCC')
    plt.tight_layout()
    plt.show()
