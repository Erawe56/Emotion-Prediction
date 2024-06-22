import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re
import nltk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.metrics import classification_report, accuracy_score

# Load the dataset
data = pd.read_csv("D:/emotion/Emotion_final.csv")

# Display the first few rows to verify column names
print(data.head())

# Visualize the distribution of emotions
sns.countplot(x=data["Emotion"], data=data, palette=["lightblue", "purple","red","green" ,"orange","yellow"])
plt.show()

# Download the stopwords dataset
nltk.download('stopwords')

# Get the list of stopwords for English
stop_words = set(stopwords.words('english'))

# Function to preprocess the text
def preprocess_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.lower()  # Convert to lowercase
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]  # Remove stopwords
    text = ' '.join(filtered_words)
    return text

# Function to count stopwords in a given text
def count_stopwords(text):
    words = text.split()
    stopword_count = sum(1 for word in words if word in stop_words)
    return stopword_count

# Preprocess the text data
data['Text'] = data['Text'].apply(preprocess_text)

# Count the stopwords
data['stopword_count'] = data['Text'].apply(count_stopwords)

# Display the DataFrame to ensure 'stopword_count' is created
print(data.head())

# Sum up all the stopword counts
total_stopwords = data['stopword_count'].sum()

print(f'Total number of stopwords in the dataset: {total_stopwords}')

# Divide the data into training and testing sets.

X = data['Text']
y = data['Emotion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert Text Data into Numerical Data
# Use techniques like TF-IDF or Count Vectorizer to convert text data into numerical vectors.

# Vectorize text data
vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# Encode the Labels
# Convert the categorical emotion labels into numeric labels.

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

# Verify the class names
print(le.classes_)  # This should output the class names as strings


# Train a Machine Learning Model
model = LogisticRegression()
model.fit(X_train_vect, y_train_encoded)

# Evaluate the Model

y_pred = model.predict(X_test_vect)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))

# Save the Model (Optional)
# Save the trained model and vectorizer for future use.

# Save the model
joblib.dump(model, 'emotion_classifier_model.pkl')

# Save the vectorizer
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')

#  Load the Model 
# Load the model and vectorizer to make predictions on new data.

# Load the model
model = joblib.load('emotion_classifier_model.pkl')

# Load the vectorizer
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# Example prediction
new_comments = ["I am  afraid "]
new_comments_tfidf = vectorizer.transform(new_comments)
predictions = model.predict(new_comments_tfidf)
predicted_emotions = le.inverse_transform(predictions)

print(predicted_emotions)














