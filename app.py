import streamlit as st
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from sklearn.metrics import accuracy_score
import pandas as pd

# Load the model and vectorizer
model = joblib.load('emotion_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Define emotion names
emotion_names = {0: 'anger', 1: 'fear', 2: 'happy',3: "love",4: "sadness",5 :"surprise" }

# # Define emotion names and corresponding emojis

emojis = {
    'anger': '😡',
    'fear': '😨',
    'happy': '😊',
    'love' : '❤️' ,
    'sadness': '😢',
    'surprise': '😱',

}

# # Load the dataset
# data = pd.read_csv("D:/emotion/Emotion_classify_Data.csv")
# X_test = data['Comment']
# y_test = data['Emotion']



# Function to preprocess text
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()  # Convert to lowercase
    return text

# # Function to predict emotion
# def predict_emotion(text):
#     text = preprocess_text(text)
#     text_vectorized = vectorizer.transform([text])
#     emotion_label = model.predict(text_vectorized)[0]
#     predicted_emotion = emotion_names.get(emotion_label, 'Unknown')
#     return predicted_emotion

# Function to predict emotion
def predict_emotion(text):
    text = preprocess_text(text)
    text_vectorized = vectorizer.transform([text])
    emotion_label = model.predict(text_vectorized)[0]
    predicted_emotion = emotion_names.get(emotion_label, 'Unknown')
    return predicted_emotion

# # Streamlit app
# def main():
#     st.title('Emotion Classifier')

#     # Text input for user input
#     user_input = st.text_input('Enter text:', '')

#     # Predict button
#     if st.button('Predict'):
#         if user_input:
#             predicted_emotion = predict_emotion(user_input)
#             st.write(f'Predicted Emotion: {predicted_emotion}')
#             # # Display accuracy
#             # y_pred = [predict_emotion(text) for text in X_test]
#             # accuracy = accuracy_score(y_test, y_pred)
#             # st.write(f'Accuracy: {accuracy:.2f}')           

#         else:
#             st.write('Please enter some text.')

  
# if __name__ == '__main__':
#     main()

# Streamlit app
def main():
    st.title('Emotion Classifier')

    # Text input for user input
    user_input = st.text_input('Enter text:', '')

    # Predict button
    if st.button('Predict'):
        if user_input:
            predicted_emotion = predict_emotion(user_input)
            emoji =emojis.get(predicted_emotion, '🤔')
            st.write(f'Predicted Emotion: {predicted_emotion} {emoji}')
        else:
            st.write('Please enter some text.')

if __name__ == '__main__':
    main()



