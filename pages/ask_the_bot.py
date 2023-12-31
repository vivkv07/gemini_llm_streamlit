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
    page_title="Ask the Bot",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.extremelycoolapp.com/help',
        'Report a bug': "https://www.extremelycoolapp.com/bug",
        'About': "# This is a header. This is an *extremely* cool app!"
    }
)
load_dotenv()

genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

def display():
    st.title("Ask Me Anything 🤖")
    st.caption("Using Gemini AI")

    # Initialize session state for storing chat history if not present
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Initialize the GEMINI model
    model = genai.GenerativeModel('gemini-pro')
    chat = model.start_chat(history=st.session_state.chat_history)

    # User input with a chat input box
    user_input = st.chat_input("Type your question here...", key="user_input", max_chars=60)

    if user_input:
        # Sending user message to the chat model
        response = chat.send_message(user_input, stream=True)
        response.resolve()

        # Update session state with the chat history
        st.session_state.chat_history = chat.history

    # Displaying the chat history in two columns
    for message in st.session_state.chat_history:
        cols = st.columns([1, 1]) if message.role == 'user' else st.columns([1, 1])

        with cols[0]:
            if message.role == 'user':
                with st.chat_message("user", avatar="🧑"):
                    st.write(message.parts[0].text)

        with cols[1]:
            if message.role == 'model':
                with st.chat_message("laddu", avatar="🤖"):
                   st.write(message.parts[0].text)

display()