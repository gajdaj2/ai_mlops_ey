import joblib
import streamlit as st


st.title("Defect Classification")


text_defektu = st.text_area("Wpisz opis defektu")

wyslij_do_modelu = st.button("Klasyfikuj")

if wyslij_do_modelu:
    model = joblib.load("defect_classifier.pkl")
    prediction = model.predict([text_defektu])
    st.text(f"Kategoria defektu: {prediction[0]}")





