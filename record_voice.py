import os
import tkinter as tk
from tkinter import simpledialog, messagebox
import pyaudio
import wave
import numpy as np
import scipy.io.wavfile
import scipy.signal
import scipy.ndimage

# Recording configuration
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
RECORD_SECONDS = 5   # Duration of each recording (seconds)
RECORDING_DIR = "D:/ChromeCoreDownloads/Python_n/ELECGLIC/France/AI_system/recording_voice_analysis/recording_dataset"   #recording dataset
AUGMENTED_DIR = "D:/ChromeCoreDownloads/Python_n/ELECGLIC/France/AI_system/recording_voice_analysis/smooth_recording"   #smoothing dataset

def record_audio(filename):
    audio = pyaudio.PyAudio()
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)

    print("Recording...")
    frames = []

    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    audio.terminate()

    wf = wave.open(filename, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(audio.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

# #三种数据增强的操作
# def augment_data(audio_file, augmented_folder):
#     rate, data = scipy.io.wavfile.read(audio_file)
    
#     # 声音平滑
#     smoothed = scipy.signal.savgol_filter(data, 11, 3)
#     augmented_file = os.path.join(augmented_folder, os.path.basename(audio_file).replace('.wav', '_smoothed.wav'))
#     scipy.io.wavfile.write(augmented_file, rate, smoothed.astype(np.int16))
    
#     # 高斯噪声
#     noise = np.random.normal(0, 0.01, data.shape)
#     noisy_data = data + noise * np.max(data)
#     augmented_file = os.path.join(augmented_folder, os.path.basename(audio_file).replace('.wav', '_noisy.wav'))
#     scipy.io.wavfile.write(augmented_file, rate, noisy_data.astype(np.int16))
    
#     # 时间缩放
#     scaled_data = scipy.ndimage.zoom(data, 1.1)
#     augmented_file = os.path.join(augmented_folder, os.path.basename(audio_file).replace('.wav', '_scaled.wav'))
#     scipy.io.wavfile.write(augmented_file, rate, scaled_data.astype(np.int16))

# Three data-enhanced operations simultaneously
def augment_data(audio_file, augmented_folder):
    rate, data = scipy.io.wavfile.read(audio_file)
    
    # Perform three operations in sequence
    # Sound smoothing
    smoothed = scipy.signal.savgol_filter(data, 11, 3)
    
    # 2. Gaussian noise
    noise = np.random.normal(0, 0.01, data.shape)
    noisy_data = smoothed + noise * np.max(smoothed)
    
    # 3. Time scaling
    scaled_data = scipy.ndimage.zoom(noisy_data, 1.1)
    
    # Save the enhanced audio file
    augmented_file = os.path.join(augmented_folder, os.path.basename(audio_file).replace('.wav', '_augmented.wav'))
    scipy.io.wavfile.write(augmented_file, rate, scaled_data.astype(np.int16))


def save_recording(name, index):
    original_folder = os.path.join(RECORDING_DIR, name)
    augmented_folder = os.path.join(AUGMENTED_DIR, name)
    if not os.path.exists(original_folder):
        os.makedirs(original_folder)
    if not os.path.exists(augmented_folder):
        os.makedirs(augmented_folder)

    filename = os.path.join(original_folder, f"{name}_{index}.wav")
    record_audio(filename)
    augment_data(filename, augmented_folder)
    messagebox.showinfo("Info", f"The recording was saved to {filename}")

def start_recording():
    name = simpledialog.askstring("Input", "speaker name:")
    if name:
        index = 1
        while True:
            record_another = messagebox.askyesno("Continue", f"Record audio at {index}?")
            if record_another:
                save_recording(name, index)
                index += 1
            else:
                break

# create GUI
root = tk.Tk()
root.title("Voice")

record_button = tk.Button(root, text="Start recording", command=start_recording)
record_button.pack(pady=20)

root.mainloop()