import pickle

import streamlit as st


with open("credit_card_approval_model.pkl", "rb") as file:
    model = pickle.load(file)


st.title("Kwalifikacja do otrzymania karty kredytowej")


st.write(""" Add customer data to check if they qualify for a credit card. """)

age = st.number_input("Customer's age", min_value=18, max_value=100, value=25, step=1)
income = st.number_input("Customer's annual income (in PLN)", min_value=10000, max_value=1000000, value=50000, step=1000)
credit_score = st.number_input("Customer's credit score", min_value=300, max_value=850, value=650, step=10)


if st.button("Check qualification"):
    features = [[age, income, credit_score]]
    prediction = model.predict(features)
    result = "Qualification accepted" if prediction[0] == 1 else "Qualification rejected"

    st.subheader("Result:")
    st.write(result)