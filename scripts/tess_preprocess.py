import os
import pandas as pd
import librosa
import matplotlib.pyplot as plt

# Define paths
TESS = "../data/tess/TESS Toronto emotional speech set data/"
output_file = "../Preprocessed Data/tess_data.csv"

# Emotion mapping
emotion_map = {
    'neutral': 'neutral', 'happy': 'happy', 'sad': 'sad',
    'angry': 'angry', 'fear': 'fear', 'disgust': 'disgust', 'pleasant_surprise': 'surprise'
}

# Process TESS dataset
def process_tess():
    tess_directory_list = os.listdir(TESS)
    file_emotion = []
    file_path = []
    file_duration = []

    for dir in tess_directory_list:
        emotion_folder = os.path.join(TESS, dir)
        if os.path.isdir(emotion_folder):
            files = os.listdir(emotion_folder)
            for file in files:
                if file.endswith(".wav"):
                    emotion = dir.split('_')[-1].lower()
                    file_emotion.append(emotion_map.get(emotion, emotion))
                    file_path.append(os.path.join(emotion_folder, file))

                    # Calculate file duration
                    duration = librosa.get_duration(path=os.path.join(emotion_folder, file))
                    file_duration.append(duration)

    return pd.DataFrame({'Emotions': file_emotion, 'Path': file_path, 'Duration': file_duration})

# Generate Graphs for PPT
def generate_graphs(df):
    # Emotion Distribution
    plt.figure(figsize=(10, 6))
    df['Emotions'].value_counts().plot(kind='bar')
    plt.title("Distribution of Emotions in TESS Dataset")
    plt.xlabel("Emotions")
    plt.ylabel("Number of Files")
    plt.grid(axis='y')
    plt.savefig("../Preprocessed Data/emotion_distribution.png")
    plt.show()

    # Duration Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['Duration'], bins=30, color='skyblue')
    plt.title("Audio File Duration Distribution")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Number of Files")
    plt.grid(axis='y')
    plt.savefig("../Preprocessed Data/audio_duration_distribution.png")
    plt.show()

if __name__ == "__main__":
    # Process TESS and Save
    tess_df = process_tess()
    os.makedirs("../Preprocessed Data/", exist_ok=True)
    tess_df.to_csv(output_file, index=False)
    print(f"TESS dataset saved to: {output_file}")

    # Generate Graphs
    generate_graphs(tess_df)
    print("Graphs saved for presentation.")
