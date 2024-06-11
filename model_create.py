import os
import cv2
import librosa
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import joblib


# Functions for extracting LBP features from PNG images
#
# def lbp_calculate(image):
#     lbp_features = []
#     for channel in range(3):
#         gray = image[:, :, channel]
#         lbp_image = np.zeros_like(gray)
#
#         for i in range(1, gray.shape[0] - 1):
#             for j in range(1, gray.shape[1] - 1):
#                 center = gray[i, j]
#                 binary_string = [
#                     gray[i - 1, j - 1] >= center,
#                     gray[i - 1, j] >= center,
#                     gray[i - 1, j + 1] >= center,
#                     gray[i, j + 1] >= center,
#                     gray[i + 1, j + 1] >= center,
#                     gray[i + 1, j] >= center,
#                     gray[i + 1, j - 1] >= center,
#                     gray[i, j - 1] >= center
#                 ]
#                 lbp_value = sum([val << idx for idx, val in enumerate(binary_string)])
#                 lbp_image[i, j] = lbp_value
#
#         lbp_features.extend(lbp_image.flatten())
#
#     return np.array(lbp_features)

#
# # wav to spectrogram-image
# def extract_spectrogram(wav_file, output_dir, person_name):
#     y, sr = librosa.load(wav_file, sr=None)
#     S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
#     S_DB = librosa.power_to_db(S, ref=np.max)
#
#     # Save spectrogram as image
#     plt.figure(figsize=(10, 4))
#     librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Mel-frequency spectrogram')
#     plt.tight_layout()
#
#     # Create output directory for the person
#     person_output_dir = os.path.join(output_dir, person_name)
#     if not os.path.exists(person_output_dir):
#         os.makedirs(person_output_dir)
#
#     # Create output file path
#     file_name = os.path.splitext(os.path.basename(wav_file))[0]
#     output_file = os.path.join(person_output_dir, f'{file_name}.png')
#
#     # Save the figure
#     plt.savefig(output_file)
#     plt.close()
#
#     return output_file
#

# load dataset
# def load_dataset(base_path, output_dir):
#     features = []
#     labels = []
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#
#     for person_name in os.listdir(base_path):
#         person_path = os.path.join(base_path, person_name)
#         if os.path.isdir(person_path):
#             for wav_file in os.listdir(person_path):
#                 wav_path = os.path.join(person_path, wav_file)
#                 spectrogram_path = extract_spectrogram(wav_path, output_dir, person_name)
#                 spectrogram_image = cv2.imread(spectrogram_path)
#                 spectrogram_resized = cv2.resize(spectrogram_image, (128, 128), interpolation=cv2.INTER_AREA).astype(
#                     np.float32)
#                 lbp_features = lbp_calculate(spectrogram_resized)
#                 features.append(lbp_features)
#                 labels.append(person_name)
#     return np.array(features), np.array(labels)


# Training models
def train_models(X_train, X_test, y_train, y_test):
    models = {
        'SVM': SVC(kernel='linear'),
        'Random Forest': RandomForestClassifier(),
        'KNN': KNeighborsClassifier(),
        'Logistic Regression': LogisticRegression(max_iter=1000),
    }

    trained_models = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{model_name} accuracy: {accuracy}")
        trained_models[model_name] = model

    return trained_models


# Main function
if __name__ == "__main__":
    base_path = 'all_lbp.txt'
    # Load LBP features from file
    features_dict = {}
    with open(base_path, 'r') as f:
        for line in f:
            parts = line.strip().split(':')
            person = parts[0]
            features_str = parts[1].split(';')[:-1]
            features = [list(map(int, feature.split(','))) for feature in features_str]
            features_dict[person] = features

    # Prepare data for training
    X, y = [], []
    for person, features in features_dict.items():
        X.extend(features)
        y.extend([person] * len(features))

    X = np.array(X)
    y = np.array(y)

    # Split data sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Train models
    models = train_models(X_train, X_test, y_train, y_test)

    # Save the trained model
    joblib.dump(models['SVM'], 'SVM_model.pkl')
    print("Model saved as SVM_model.pkl")
    joblib.dump(models['Random Forest'], 'Random_Forest_model.pkl')
    print("Model saved as Random_Forest_model.pkl")
    joblib.dump(models['KNN'], 'KNN_model.pkl')
    print("Model saved as KNN_model.pkl")
    joblib.dump(models['Logistic Regression'], 'Logistic_Regression_model.pkl')
    print("Model saved as Logistic_Regression_model.pkl")
