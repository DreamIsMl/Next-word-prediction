import pickle

import numpy as np
import streamlit as st
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the tokenizer from the pkl file
# Replace 'path/to/your/tokenizer.pkl' with the actual path
with open('F:\\class\\DeepLearning\\Neext word prediction\\token.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Load your model
# Replace 'path/to/your/model' with the actual path
model = load_model('F:\\class\\DeepLearning\\Neext word prediction\\ln.h5')

def predict_next_words(sentence, top_n=3):
    # Tokenize
    token_text = tokenizer.texts_to_sequences([sentence])[0]
    # Padding
    padded_token_text = pad_sequences([token_text], maxlen=151, padding='pre')
    # Predict probabilities for each word
    probabilities = model.predict(padded_token_text)[0]
    # Get indices of the top N predicted words
    top_indices = np.argsort(probabilities)[-top_n:][::-1]

    # Find the words in English
    predicted_words = [word for word, index in tokenizer.word_index.items() if index in top_indices]

    return predicted_words

# Streamlit app
st.title("Word Prediction App")

# Input text
user_input = st.text_input("Enter a sentence:")

# Predict button
if st.button("Predict"):
    # Predict the next words
    next_words = predict_next_words(user_input, top_n=5)

    # Display predicted next words
    st.write("Predicted Next Words:", next_words)
