import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the pre-trained model and vectorizer
model = joblib.load(r'D:/MY AI PROJECTS/Good or Bad Classification/Machine Learning Model/svm_review_classification_model.h5')

vectorizer = joblib.load(r'D:/MY AI PROJECTS/Good or Bad Classification/Machine Learning Model/tfidf_vectorizer.pkl')  # Make sure to save and load the vectorizer

# Function to predict the review sentiment
def predict_review(review):
    # Vectorize the input review
    review_vectorized = vectorizer.transform([review])
    prediction = model.predict(review_vectorized)
    return prediction[0]  # Return the predicted class

# Streamlit app
def main():
    st.title("Review Classification App")
    st.write("Type a review to classify it as good or bad.")

    # User input for the review
    user_input = st.text_area("Enter your review here:")

    # Make prediction when button is clicked
    if st.button("Predict"):
        if user_input:
            result = predict_review(user_input)
            st.write("Predicted Review Sentiment:", "Good" if result == 1 else "Bad")
        else:
            st.write("Please enter a review.")

if __name__ == "__main__":
    main()


