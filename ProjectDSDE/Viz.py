import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
from sklearn.preprocessing import LabelEncoder
import re
import string
import pickle

# Function to preprocess the input data
def preprocess_input(book_info):
    def clean_text(text):  
        text = re.sub(r"[\[\]']", "", str(text))
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    book_info = clean_text(book_info)

    def normalize_text(text):
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans("", "", string.punctuation))
        return text
     
    book_info = normalize_text(book_info)

    # Preprocess the book information (e.g., title, abstract)
    vectorizer = tf.keras.layers.TextVectorization(max_tokens=1000, output_sequence_length=100)
    vectorizer.adapt(np.array([book_info]))
    new_input_vectorized = vectorizer(np.array([book_info]))  # Wrap in np.array to match expected input format

    return np.array(new_input_vectorized)

# Function to predict the subject
def predict_subject(publich_name, organizations, classifications, auth_keywords):
    feature_string = organizations + " classifications: " + classifications + \
                      " publish_name: " + publich_name + " auth-keywords: " + auth_keywords

    # Preprocess the book information
    processed_input = preprocess_input(feature_string)
    
    # Predict the probabilities for each subject
    predictions = model.predict(processed_input)
    
    # Get the index of the highest probability
    best_subject_index = np.argmax(predictions)
    
    # Map index to class name (your subject labels)
    predicted_subject = label_encoder.inverse_transform([best_subject_index])[0]
    
    return predicted_subject


tf.keras.backend.clear_session()
# Load the model
model = load_model('my_dnn_model.keras')

# Load the LabelEncoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)

# Streamlit Layout
st.title("Research Subject Prediction & Book Recommendation Dashboard")

# Input Section
st.sidebar.header("Input for Prediction")
book_publich_name = st.sidebar.text_input("Enter Publish Name")
book_organizations = st.sidebar.text_input("Enter Organizations")
book_classifications = st.sidebar.text_input("Enter Classifications (you can input more than 1 classification)")
auth_keywords = st.sidebar.text_input("Enter Author Keywords (you can input more than 1 keyword)")

# Select subject filter
subject_list = ['Physics and Astronomy', 'Chemistry', 'Engineering', 'Computer Science and Information Technology', 
                'Biology', 'Medicine', 'Agriculture and Food Science', 'Social Sciences', 'Humanities']
selected_subject = st.sidebar.selectbox("Select Subject", subject_list, index=0)  # Default to first subject

if st.sidebar.button("Predict Subject"):
    if not (book_publich_name and book_organizations and book_classifications and auth_keywords):
        st.sidebar.error("Please fill all fields before prediction.")
        predicted_subject = None
    else:
        predicted_subject = predict_subject(book_publich_name, book_organizations, book_classifications, auth_keywords)
        if predicted_subject:
            st.sidebar.success(f"Predicted Subject: {predicted_subject}")
            selected_subject = predicted_subject  # Set the filter to the predicted subject
else:
    predicted_subject = None

# Visualization Section
if selected_subject:
    st.header(f"Books recommended for Subject: {selected_subject}")

    try:
        books_df = pd.read_csv(f'{selected_subject}.csv')  # Assuming the file is named after the subject
        st.write(f"Displaying books for subject: {selected_subject}")
        
        # Display total books count
        total_books = len(books_df)
        st.write(f"Total Books: {total_books}")

        # Display the table with book details
        st.dataframe(books_df[['title', 'author_name', 'href', 'ratings_avg', 'ratings_count', 'readinglog_count']])

        # You can also display some charts for the data
        st.subheader("Book Ratings Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(books_df['ratings_avg'], bins=20, color='skyblue', edgecolor='black')
        ax.set_title("Distribution of Average Ratings")
        ax.set_xlabel("Average Rating")
        ax.set_ylabel("Number of Books")
        st.pyplot(fig)

        st.subheader("Book Views Distribution")
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(books_df['readinglog_count'], bins=20, color='orange', edgecolor='black')
        ax.set_title("Distribution of Book Views")
        ax.set_xlabel("Number of Views")
        ax.set_ylabel("Number of Books")
        st.pyplot(fig)

    except FileNotFoundError:
        st.error(f"No CSV file found for subject: {selected_subject}")
