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
    page_title="GemLit - Content Wizard",
    page_icon="🧊",
    initial_sidebar_state="expanded"
)
with st.sidebar:
    st.caption("Developer: Vivek Kv")
    st.markdown('''[![Streamlit App](https://badgen.net/pypi/v/streamlit)](https://pypi.org/project/streamlit/)
                [![Linkedin](https://flat.badgen.net/badge/linkedin//connect?icon=linkedin)](https://www.linkedin.com/in/vivekkovvuru/)''')
    st.info("Code and Tutorials will be shared on Linkedin!")
    st.markdown("[![BymeaCoffee](https://badgen.net/badge/icon/buymeacoffee?icon=buymeacoffee&label)](https://www.buymeacoffee.com/vivekkovvuru)")


load_dotenv()
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

def to_markdown(text):
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

def display():
    st.title("Let's write some Content!")
    st.caption("Powered with Google Gemini AI")

    user_input = st.text_input("Input your topic of interest here.", max_chars=120)
    user_input = "You are a professional Content Writer. Write an article on "+ user_input
    if st.button('Submit') and user_input:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(user_input)
        st.markdown(to_markdown(response.text), unsafe_allow_html=True)

display()
    