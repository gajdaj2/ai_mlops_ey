import joblib
import streamlit as st
import vectorizer

st.title("Movie Genre Classification")

comment = st.text_area("Enter your comment")

classify = st.button("Classify")

if classify:
    st.write("Classifying...")
    # Load the model
    model = joblib.load("movie_review_classifier.pkl")

    new_comments_tfidf = vectorizer.transform(comment)
    new_predictions = model.predict(new_comments_tfidf)
    print("\nPredictions for new comments:")
    for comment, classification in zip(comment, new_predictions):
        print(f"'{comment}' -> {classification}")
    st.write(f"Predicted genre: {new_predictions[0]}")
