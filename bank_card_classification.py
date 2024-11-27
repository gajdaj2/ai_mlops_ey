# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Przygotowanie danych
# Przykładowe dane (zastąp je rzeczywistymi danymi)
data = {
    "age": [25, 45, 35, 50, 23],
    "income": [50000, 80000, 60000, 120000, 40000],
    "credit_score": [650, 700, 800, 750, 600],
    "approved": [0, 1, 1, 1, 0]  # 0 = accepted, 1 = not accepted
}

df = pd.DataFrame(data)

# 2. Przygotowanie zmiennych X i y
X = df[["age", "income", "credit_score"]]
y = df["approved"]

# 3. Podział na dane treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Trenowanie modelu
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 5. Ocena modelu
y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# 6. Zapisanie modelu do pliku
with open("credit_card_approval_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model zapisany do pliku: credit_card_approval_model.pkl")