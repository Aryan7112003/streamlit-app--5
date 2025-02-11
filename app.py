import streamlit as st
import pandas as pd
import joblib
import numpy as np
import re
import string
import random  # For random selection
from faker import Faker  # For generating random names

# Load the trained Random Forest model and TF-IDF vectorizer
try:
    model = joblib.load('random_forest_model.pkl')  # Ensure this file exists
    vectorizer = joblib.load('tfidf_vectorizer.pkl')  # Ensure this file exists
except FileNotFoundError:
    st.error("Error: Model or vectorizer file not found! Make sure they are in the same directory.")

# Function to clean and preprocess text
def clean_text(text):
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\d+', '', text)  # Remove numbers
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = text.strip()  # Remove extra spaces
    
    # Handling negation words
    negations = {"isn't": "is not", "wasn't": "was not", "don't": "do not", "didn't": "did not",
                 "won't": "will not", "shouldn't": "should not", "couldn't": "could not",
                 "can't": "cannot", "weren't": "were not", "doesn't": "does not"}
    for neg, full_form in negations.items():
        text = text.replace(neg, full_form)
    
    return text

# Load CSV file containing reviews (Real user names will be hidden)
csv_path = "E:/NLP(Sentimate analysis project)/245_1.csv"  # Corrected file path
df = pd.read_csv(csv_path)

# Create a Faker instance to generate random fake names
fake = Faker()

# Ask user to input a review
st.title("Sentiment Analysis of Product Reviews")
st.write("Enter a product review below, and the app will predict its sentiment and show a fake user name.")

# Input box for review text
user_review = st.text_area("Enter Review:")

# Button to predict sentiment
if st.button("Predict Sentiment"):
    if user_review.strip():  # Check if input is not empty
        # Clean and preprocess the review
        cleaned_review = clean_text(user_review)
        st.write("Cleaned Review: ", cleaned_review)  # Debug: Show cleaned review
        
        # Convert cleaned text to vector using the TF-IDF vectorizer
        review_vector = vectorizer.transform([cleaned_review])  # Convert text to vector
        st.write("Review Vector: ", review_vector.toarray())  # Debug: Show vector
        
        # Predict sentiment
        prediction = model.predict(review_vector)  # Predict sentiment
        probabilities = model.predict_proba(review_vector)  # Get prediction probabilities
        
        st.write("Prediction Probabilities: ", probabilities)  # Debug: Show prediction probabilities
        
        confidence = np.max(probabilities) * 100  # Convert to percentage
        st.write("Prediction Confidence: ", confidence)  # Debug: Show confidence
        
        # Generate a random fake name
        fake_name = fake.name()

        # Check if the generated name is more likely male or female
        name_gender = "Male" if any(male_name in fake_name for male_name in ["John", "Mike", "David", "James", "Robert"]) else "Female"

        # Display appropriate emoji based on the name gender
        if name_gender == "Male":
            emoji = "👨"  # Man emoji
        else:
            emoji = "👩"  # Woman emoji

        # Create a DataFrame to display in table format
        result_df = pd.DataFrame({
            "User Name": [fake_name + " " + emoji],  # Show the fake name with emoji
            "Review": [user_review],
            "Predicted Sentiment": [prediction[0]],
            "Confidence Score": [f"{confidence:.2f}%"]
        })

        # Display results in table format
        st.write("### Review Prediction Table:")
        st.table(result_df)  # Display as table

    else:
        st.write("⚠️ Please enter a review to analyze.")
