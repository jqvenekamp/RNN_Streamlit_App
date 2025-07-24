import streamlit as st
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
import numpy as np

# Recompile the model and model architecture
vocab_size = 10000  # Adjust based on your dataset
max_length = 500  # Adjust based on your dataset
model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
    SimpleRNN(32, activation='relu', return_sequences=False),
    Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])  
model.build(input_shape=(None, max_length))
model.load_weights('simplernn_model.h5')  # Load the weights from the saved model

# Load the word index
word_index = imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}

# Function to preprocess input text
def preprocess_text(text):
    text = text.replace('!', '').replace('.', '').replace(',', '')
    tokens = text.lower().split()
    token_ids = [word_index.get(token, 2) + 3 for token in tokens]
    return sequence.pad_sequences([token_ids], maxlen=500)

# Streamlit app
st.title("Sentiment Analysis with RNN")
user_input = st.text_area("Enter your text here:")
if st.button("Predict"):
    if user_input:
        preprocessed_input = preprocess_text(user_input)
        prediction = model.predict(preprocessed_input)
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
        confidence = prediction[0][0] if sentiment == "Positive" else 1 - prediction[0][0]
        st.write(f"Sentiment: {sentiment} (Confidence: {confidence:.2f})")
    else:
        st.error("Please enter some text to analyze.")