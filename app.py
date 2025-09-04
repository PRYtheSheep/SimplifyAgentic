import streamlit as st
import webbrowser

if "browser_opened" not in st.session_state:
    webbrowser.open_new("http://localhost:8501")
    st.session_state.browser_opened = True

st.title("Simple Streamlit Demo")

name = st.text_input("Enter your name:")
age = st.number_input("Enter your age:", min_value=0, max_value=120, step=1)
rating = st.slider("How do you rate this demo?", 0, 10, 5)

if st.button("Submit"):
    st.success(f"Hello {name}! You are {age} years old and rated this demo {rating}/10.")

if st.checkbox("Show secret message"):
    st.info("🎉 Streamlit is fun and easy to use!")
