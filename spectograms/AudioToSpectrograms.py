import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
audio_directory = "C:/Users/INFOKOM/Desktop/stage_pfe/baby_cries_classification/data"  
output_directory = "C:/Users/INFOKOM/Desktop/stage_pfe/baby_cries_classification/spectrograms"
# Function to create spectrograms
def create_spectrogram(file_path, output_path):
    y, sr = librosa.load(file_path, sr=22050)
    plt.figure(figsize=(5, 5))
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Loop through dataset and generate spectrograms
for label in os.listdir(audio_directory):
    label_path = os.path.join(audio_directory, label)
    output_label_path = os.path.join(output_directory, label)
    os.makedirs(output_label_path, exist_ok=True)

    if os.path.isdir(label_path):
        for filename in os.listdir(label_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(label_path, filename)
                output_path = os.path.join(output_label_path, filename.replace('.wav', '.png'))
                create_spectrogram(file_path, output_path)

print("Spectrograms generated successfully!")
