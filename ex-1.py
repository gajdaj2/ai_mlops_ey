import random

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Sample comments and classifications
comments = [
    "Amazing movie, really loved it!",
    "Terrible film, waste of time.",
    "The acting was fantastic, but the plot was weak.",
    "Great visuals but the story was a bit slow.",
    "I would watch it again, highly recommended!",
    "Not my cup of tea, but some might like it.",
    "A masterpiece! Stunning performances.",
    "Predictable and boring.",
    "Absolutely thrilling, kept me on the edge of my seat.",
    "Disappointed, expected much better.",
    "Unique storyline, very refreshing.",
    "Poorly written, felt rushed.",
    "Loved the cinematography, breathtaking!",
    "An okay film, nothing special.",
    "Horrible acting, couldn't finish it."
]

# Classifications: Positive, Neutral, Negative
classifications = ["Positive", "Neutral", "Negative"]

# Generate dataset
num_samples = 50
data = {
    "Comment": [random.choice(comments) for _ in range(num_samples)],
    "Classification": [random.choice(classifications) for _ in range(num_samples)]
}

comments_df = pd.DataFrame(data)
print(comments_df.head())


df = pd.DataFrame(data)

X = df["Comment"]  # Komentarze
y = df["Classification"]  # Klasyfikacja

# Podział danych na zestawy treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Przetwarzanie tekstu: TF-IDF Wektoryzacja
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Budowa modelu klasyfikacyjnego
model = RandomForestClassifier(random_state=42)
model.fit(X_train_tfidf, y_train)

# Prognozowanie
y_pred = model.predict(X_test_tfidf)

# Ewaluacja modelu
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Przykładowe przewidywanie
new_comments = [
    "I loved the movie, it was fantastic!",
    "The film was terrible, I hated it.",
    "It was an okay movie, nothing special."
]

new_comments_tfidf = vectorizer.transform(new_comments)
new_predictions = model.predict(new_comments_tfidf)
print("\nPredictions for new comments:")
for comment, classification in zip(new_comments, new_predictions):
    print(f"'{comment}' -> {classification}")

joblib.dump(model, "movie_review_classifier.pkl")

