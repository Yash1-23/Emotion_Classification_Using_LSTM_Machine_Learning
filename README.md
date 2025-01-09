# Emotion_Detection_Using_LSTM_Machine_Learning

## This project implements an Emotion Detection System that uses a Long Short-Term Memory (LSTM) model to predict emotions based on input text. It leverages deep learning techniques alongside natural language processing (NLP) to analyze the sentiment and classify it into predefined emotion categories.

## Features
Text Preprocessing: Cleans and prepares raw input text using NLP techniques such as stemming and stopword removal.
Deep Learning: Employs an LSTM model for capturing sequential dependencies in text data.
Interactive UI: Provides a user-friendly interface for emotion prediction using either Flask or Streamlit.
Pre-trained Components: Includes pre-trained models and vectorizers for easy integration and use.

## Run the Streamlit application:
streamlit run app.py


## How It Works
1.Text Input: User inputs text (e.g., "I am feeling so happy today").
2.Text Preprocessing:
  Removes non-alphabetic characters.
  Converts text to lowercase.
  Removes stopwords and applies stemming.
3.Feature Extraction: Converts text into a numerical representation using TF-IDF.
4.Emotion Prediction:
  The LSTM model processes the numerical features.
  Outputs the predicted emotion label (e.g., Happy, Sad, Angry, etc.).
5.Display Result: Shows the predicted emotion to the user.

## Model Details
Architecture: LSTM
Input Features: TF-IDF vectorized text data.
Training Dataset: The model was trained on a labeled emotion dataset with categories such as:
Happy
Sad
Angry
Fear
Neutral
Preprocessing:
Tokenization
Stopword removal
Stemming
