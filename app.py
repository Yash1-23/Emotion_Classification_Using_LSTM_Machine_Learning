import streamlit as st
import pickle
import re
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import numpy as np

# Download NLTK stopwords
nltk.download('stopwords')
stopwords = set(stopwords.words('english'))

# ======================== Load Saved Files =======================================
lg = pickle.load(open('logistic_regression.pkl', 'rb'))
tfidf_vectorizer = pickle.load(open('tfidfvectorizer.pkl', 'rb'))
lb = pickle.load(open('label_encoder.pkl', 'rb'))

# ======================== Text Cleaning Function ===================================
def clean_text(text):
    stemmer = PorterStemmer()
    text = re.sub("[^a-zA-Z]", " ", text)
    text = text.lower()
    text = text.split()
    text = [stemmer.stem(word) for word in text if word not in stopwords]
    return " ".join(text)

# ======================== Predict Emotion Function ================================
def predict_emotion(input_text):
    cleaned_text = clean_text(input_text)
    input_vectorized = tfidf_vectorizer.transform([cleaned_text])
    predicted_label = lg.predict(input_vectorized)[0]
    predicted_emotion = lb.inverse_transform([predicted_label])[0]
    return predicted_emotion

# ======================== Streamlit User Interface ===============================
st.title("Emotion Detection System")

# Input Text
st.write("Enter a sentence to detect the emotion:")
input_text = st.text_area("Type your text here...", height=100)

# Button to Predict
if st.button("Analyze Emotion"):
    if input_text.strip():
        predicted_emotion = predict_emotion(input_text)
        st.success(f"Predicted Emotion: **{predicted_emotion}**")
    else:
        st.error("Please enter some text to analyze.")

