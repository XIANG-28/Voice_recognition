import os
import cv2
import numpy as np


# Function for extracting LBP features from PNG images
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
                lbp_image[i, j] = int(round(lbp_value))
        lbp_features.extend(lbp_image.flatten())
    return np.array(lbp_features)


def load_dataset(bath_path, output_dir):
    features_dict = {}
    for person_name in os.listdir(bath_path):
        person_path = os.path.join(bath_path, person_name)
        if os.path.isdir(person_path):
            person_features = []
            for image_file in os.listdir(person_path):
                spectrogram_path = os.path.join(person_path, image_file)
                spectrogram_image = cv2.imread(spectrogram_path)
                spectrogram_resized = cv2.resize(spectrogram_image, (128, 128), interpolation=cv2.INTER_AREA).astype(
                    np.float32)
                lbp_features = lbp_calculate(spectrogram_resized)
                person_features.append(lbp_features)
            features_dict[person_name] = person_features

    # Save LBP features to file
    with open(output_dir, 'w') as f:
        for person, features in features_dict.items():
            f.write(f"{person}:")
            for feature in features:
                f.write(','.join(map(str, map(int, feature))))  #特征值转换成整数，再保存
                f.write(';')
            f.write('\n')

if __name__ == "__main__" :
    image_path = 'image'
    output_dir='all_lbp.txt'
    load_dataset(image_path,output_dir)