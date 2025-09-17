import streamlit as st
import pickle

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

st.title("📰 Fake News Detector")
st.write("Paste a news article to check if it's **Fake** or **Real**.")

user_input = st.text_area("Enter news article here:")

if st.button("Predict"):
    if user_input.strip():
        input_vector = vectorizer.transform([user_input])
        prediction = model.predict(input_vector)[0]
        result = "✅ Real News" if prediction == 1 else "❌ Fake News"
        st.success(result)
    else:
        st.warning("Please enter some text before predicting.")
