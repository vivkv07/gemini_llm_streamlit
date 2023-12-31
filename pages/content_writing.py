import streamlit as st
import textwrap
import base64
from PIL import Image
from io import BytesIO
import google.generativeai as genai
import google.ai.generativelanguage as glm
import os
from dotenv import load_dotenv
import google.generativeai as genai

st.set_page_config(
    page_title="Write Articles",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

load_dotenv()
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

def to_markdown(text):
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

def display():
    st.title("Let's write some Content!")
    user_input = st.text_input("Input your topic of interest here.", max_chars=120)
    user_input = "You are a professional Content Writer. Write an article on "+ user_input
    if st.button('Submit') and user_input:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(user_input)
        st.markdown(to_markdown(response.text), unsafe_allow_html=True)

display()
    