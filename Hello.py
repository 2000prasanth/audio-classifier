import streamlit as st
import librosa
import numpy as np
import pickle
from tensorflow.keras.models import load_model
import joblib
import resampy

# Load the label encoder
labelencoder = joblib.load('labelencoder.pkl')

# Load the model from the SavedModel format
#loaded_model = tf.saved_model.load(r'C:\Users\prasa\OneDrive\Desktop\DSA hackathons\AUdio DL\saved_models\audio_classification.hdf5')

model = load_model('audio_classification.hdf5')
# Function to preprocess audio file
def preprocess_audio(filename):
    audio, sample_rate = librosa.load(filename, res_type='kaiser_fast') 
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features.reshape(1, -1)

# Define Streamlit app
def main():
    st.title('Audio Classification')

    uploaded_file = st.file_uploader("Upload an audio file", type=['wav'])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        if st.button('Classify'):
            # Preprocess the uploaded file
            preprocessed_features = preprocess_audio(uploaded_file)

            # Apply the model to classify the audio file
            predicted_label = model.predict(preprocessed_features)

            # Convert predicted label to class
            predicted_class_index = np.argmax(predicted_label)
            prediction_class = labelencoder.inverse_transform(predicted_class_index.reshape(1, -1))

            st.write("Predicted class:", prediction_class)

# Run the Streamlit app
if __name__ == '__main__':
    main()
