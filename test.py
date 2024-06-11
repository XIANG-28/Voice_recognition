import os
import cv2
import numpy as np
import librosa
import pyaudio
import wave
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from sklearn.linear_model import LogisticRegression
import joblib

# Functions for extracting LBP features from PNG images

def lbp_calculate(image):
    lbp_features = []
    for channel in range(3):
        gray = image[:, :, channel]
        lbp_image = np.zeros_like(gray)

        for i in range(1, gray.shape[0] - 1):
            for j in range(1, gray.shape[1] - 1):
                center = gray[i, j]
                binary_string = [
                    gray[i-1, j-1] >= center,
                    gray[i-1, j] >= center,
                    gray[i-1, j+1] >= center,
                    gray[i, j+1] >= center,
                    gray[i+1, j+1] >= center,
                    gray[i+1, j] >= center,
                    gray[i+1, j-1] >= center,
                    gray[i, j-1] >= center
                ]
                lbp_value = sum([val << idx for idx, val in enumerate(binary_string)])
                lbp_image[i, j] = lbp_value

        lbp_features.extend(lbp_image.flatten())

    return np.array(lbp_features)

# wav to spectrogram-image
def extract_spectrogram(wav_file, output_dir):
    y, sr = librosa.load(wav_file, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)

    # Save spectrogram as image
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Create output file path
    output_file = os.path.join(output_dir, 'temp.png')

    # Save the figure
    plt.savefig(output_file)
    plt.close()

    return output_file

# Function to record audio
def record_audio(output_file, duration=5, sample_rate=44100, channels=1, chunk=1024, format=pyaudio.paInt16):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=format,
                        channels=channels,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk)
    frames = []

    print("Recording...")
    for i in range(0, int(sample_rate / chunk * duration)):
        data = stream.read(chunk)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(output_file, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(audio.get_sample_size(format))
    wf.setframerate(sample_rate)
    wf.writeframes(b''.join(frames))
    wf.close()

# Function to predict speaker from recorded audio
def predict_speaker(audio_file, model):
    # Extract spectrogram image
    spectrogram_path = extract_spectrogram(audio_file, 'D:/ChromeCoreDownloads/Python_n/ELECGLIC/France/AI_system/recording_voice_analysis/real_record')
    spectrogram_image = cv2.imread(spectrogram_path)
    spectrogram_resized = cv2.resize(spectrogram_image, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)

    # Extract LBP features
    lbp_features = lbp_calculate(spectrogram_resized)

    # Predict using trained model
    prediction = model.predict([lbp_features])
    return prediction[0]

# Main function
if __name__ == "__main__":
    # Load the trained model
    model = joblib.load('Random_Forest_model.pkl')
    print("Model loaded from Random_Forest_model.pkl")

    while True:
        # Record and predict
        record_audio('temp.wav')  # Record audio to a temporary file
        speaker = predict_speaker('temp.wav', model)
        print(f"Currently speaking: {speaker}")