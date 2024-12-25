from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import librosa
import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # Suppress unnecessary warnings

# Load pre-trained Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# Paths to input/output files
data_path = "../Preprocessed Data/tess_data.csv"  # Input CSV from TESS preprocessing
output_path = "../Extracted Features/tess_wav2vec_features.csv"

# Ensure output directory exists
os.makedirs("../Extracted Features", exist_ok=True)

# Function to extract Wav2Vec2 features
def extract_wav2vec_features(file_path):
    try:
        audio, sr = librosa.load(file_path, sr=16000)  # Ensure 16kHz sampling rate
        inputs = processor(audio, return_tensors="pt", sampling_rate=16000, padding=True)
        with torch.no_grad():
            features = model(**inputs).last_hidden_state
        return features.mean(dim=1).squeeze().numpy()  # Apply mean pooling
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Load preprocessed TESS data
df = pd.read_csv(data_path)

# Initialize lists to store features and labels
features = []
labels = []

# Process each audio file in the dataset
print("Extracting features...")
for _, row in tqdm(df.iterrows(), total=len(df), desc="Wav2Vec Feature Extraction"):
    file_path = row['Path']
    emotion = row['Emotions']
    feature = extract_wav2vec_features(file_path)
    if feature is not None:
        features.append(feature)
        labels.append(emotion)

# Save extracted features and labels
features_df = pd.DataFrame(features)
features_df['label'] = labels
features_df.to_csv(output_path, index=False)
print(f"Features saved to: {output_path}")

# Visualization: Plot feature distribution
def plot_feature_distribution(features_df):
    plt.figure(figsize=(12, 6))
    feature_means = features_df.iloc[:, :-1].mean(axis=1)
    plt.hist(feature_means, bins=50, color='blue', alpha=0.7)
    plt.title("Wav2Vec2 Feature Distribution for TESS Dataset")
    plt.xlabel("Mean Feature Value")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig("../Extracted Features/tess_feature_distribution.png")
    plt.show()

# Generate and save the plot
plot_feature_distribution(features_df)
print("Feature distribution plot saved as 'tess_feature_distribution.png'.")
