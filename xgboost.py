import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('IMDB Dataset.csv')

# Preprocess text
stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    words = nltk.word_tokenize(text)
    words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(words)

data['review'] = data['review'].apply(preprocess_text)

# Convert text data to TF-IDF features
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(data['review'])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, data['sentiment'], test_size=0.2, random_state=42)

# Train the XGBoost classifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Make predictions and evaluate the model
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Function to predict sentiment
def predict_sentiment(review):
    review = preprocess_text(review)
    review_vectorized = tfidf_vectorizer.transform([review])
    prediction = classifier.predict(review_vectorized)
    return prediction[0], accuracy

# Example usage:
example_review = "This movie was great! I loved the plot and the acting."
sentiment, accuracy = predict_sentiment(example_review)
print(f"Predicted Sentiment: {sentiment}, Model Accuracy: {accuracy}")
