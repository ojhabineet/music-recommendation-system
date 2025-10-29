🎵 Music Recommendation System using CNN and Spectrograms
📘 Overview

This project is a Music Recommendation System that leverages Convolutional Neural Networks (CNNs) and audio spectrogram analysis to classify songs based on genre, mood, or style. Using raw audio datasets, the data was first categorized and preprocessed into visual spectrograms, enabling the CNN model to learn sound features visually — similar to how image models recognize patterns.

Once trained, the system can recommend similar songs to a given input based on their learned audio feature embeddings or genre predictions.

🧠 Project Description
🎯 Objective

To develop an intelligent recommendation system that understands music not just by metadata (like artist or title), but by the actual sound characteristics — rhythm, pitch, tone, and harmony — using deep learning.

🔍 Approach

Dataset Preparation

Raw .wav or .mp3 files were collected and organized by genre.

The dataset was cleaned and categorized into folders per genre (e.g., rock/, classical/, jazz/, pop/, etc.).

Audio files were trimmed and normalized to a consistent duration and sample rate.

Feature Extraction

Audio signals were converted to Mel Spectrograms using librosa.

Spectrograms were saved as images and served as inputs to the CNN.

Each spectrogram visually represents the energy and frequency distribution of the music over time.

Model Architecture

A Convolutional Neural Network (CNN) was built using TensorFlow/Keras.

The CNN learns spatial patterns in the spectrograms that correspond to genre or mood features.

The trained embeddings were later used for song similarity and recommendation.

Recommendation System

After classification, the system computes cosine similarity between songs' CNN feature vectors.

Recommendations are generated based on closest similarity scores, helping users discover songs that sound alike.

🧩 Tech Stack
Category	Tools / Libraries
Programming Language	Python
Deep Learning	TensorFlow, Keras
Audio Processing	Librosa, NumPy
Visualization	Matplotlib, Seaborn
Data Handling	Pandas, Scikit-learn
Similarity Metric	Cosine Similarity
Environment	Jupyter / Kaggle Notebook
📂 Dataset

Dataset Type: Raw audio files (.wav / .mp3)

Source: GTZAN Dataset - Music Genre Classification
 (or replace with your actual dataset link)

Classes: Example genres — Classical, Jazz, Rock, Pop, Blues, Hip-Hop, etc.

Preprocessing:

Audio normalization and trimming

Mel-spectrogram conversion using Librosa

Saved as 2D spectrogram images for CNN input

⚙️ Model Architecture
Model: "Sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
Conv2D(32, (3,3), activation='relu')  → (None, 126, 126, 32)
MaxPooling2D((2,2))
Conv2D(64, (3,3), activation='relu')
MaxPooling2D((2,2))
Flatten()
Dense(128, activation='relu')
Dropout(0.3)
Dense(num_classes, activation='softmax')
=================================================================

🚀 How to Run
1️⃣ Clone the Repository
git clone https://github.com/yourusername/music-recommendation-system.git
cd music-recommendation-system

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Prepare the Dataset

Organize raw audio files into subfolders (one folder per category).
Example:

dataset/
├── rock/
├── classical/
├── jazz/
├── pop/


The script will automatically generate spectrograms during preprocessing.

4️⃣ Train the Model
python train_model.py

5️⃣ Generate Recommendations
python recommend.py

📊 Results

Model Accuracy: ~85–90% (depending on dataset size and preprocessing)

Loss Function: Categorical Crossentropy

Optimizer: Adam

Evaluation Metric: Accuracy

Example Spectrogram Visualization:
(insert image here if available)


💡 Future Improvements

Implement transfer learning using pre-trained CNNs like VGGish or YAMNet

Integrate lyrics and metadata for hybrid recommendations

Build a web app interface for interactive song suggestions

Deploy using Streamlit or Flask API

🧑‍💻 Author

Bineet Ojha
🎓 Deep Learning Enthusiast | 🎧 Music + AI
📫 [Your Email or LinkedIn Profile]
