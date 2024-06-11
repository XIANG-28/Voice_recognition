import librosa
import matplotlib.pyplot as plt
import os
import numpy as np


# wav to spectrogram-image
def extract_spectrogram(wav_file, output_dir, person_name):
    y, sr = librosa.load(wav_file, sr=None)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)

    # Save spectrogram as image
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-frequency spectrogram')
    plt.tight_layout()

    # Create output directory for the person
    person_output_dir = os.path.join(output_dir, person_name)
    if not os.path.exists(person_output_dir):
        os.makedirs(person_output_dir)

    # Create output file path
    file_name = os.path.splitext(os.path.basename(wav_file))[0]
    output_file = os.path.join(person_output_dir, f'{file_name}.png')

    # Save the figure
    plt.savefig(output_file)
    plt.close()

    return output_file


# load dataset
def load_dataset(base_path, output_dir):
    features = []
    labels = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for person_name in os.listdir(base_path):
        person_path = os.path.join(base_path, person_name)
        if os.path.isdir(person_path):
            for wav_file in os.listdir(person_path):
                wav_path = os.path.join(person_path, wav_file)
                extract_spectrogram(wav_path, output_dir, person_name)

if __name__ == "__main__":
    base_path = 'D:/ChromeCoreDownloads/Python_n/ELECGLIC/France/AI_system/recording_voice_analysis/wav_voice'
    image_path = 'D:/ChromeCoreDownloads/Python_n/ELECGLIC/France/AI_system/recording_voice_analysis/image'
    load_dataset(base_path, image_path)

