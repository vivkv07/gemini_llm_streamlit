import streamlit as st
# from gemini_llm.pages import ask_the_bot, content_writing
# from pages import understand_your_image, content_generation, document_query, home
from dotenv import load_dotenv
import google.generativeai as genai
import os
from streamlit_extras.app_logo import add_logo
from streamlit_toggle import st_toggle_switch
from streamlit_extras.badges import badge
from streamlit_extras.mention import mention
from streamlit_extras.metric_cards import style_metric_cards
from streamlit_elements import elements, mui, html
from streamlit_elements import nivo
# from streamlit_extras.buy_me_a_coffee import button
st.set_page_config(
    page_title="GemLit - AI assistant",
    page_icon="🧊",
    layout="wide",
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

# if __name__ == "__main__":
#     main()
def display():
    st.title("GemLit: AI Assistant")
    st.caption("Powered with Google Gemini AI")
    st.markdown("""
Welcome to **GemLit**, where the power of Gemini AI is seamlessly integrated into a user-friendly interface to assist you with a diverse array of tasks. Exploring quickest way to integrate power of AI within your applications.
[Get your API Key ](https://ai.google.dev/) 

## Core Features

- **Ask the Bot**: Engage with the AI chatbot for insightful and informative answers. Whether it's a complex query or a simple question, it will assist you.
- **Image Insight**: Upload any image for comprehensive analysis or to generate captivating stories. Dive deep into the visual world with state-of-the-art image understanding technology.
- **Content Wizard**: Transform your ideas into well-articulated content. Content writing assistance, ensuring your text is both engaging and informative.
- **Document Intellect**: Easily upload and sift through PDF documents. Using ChromaDB for Vector Collections, enables you to find precise information swiftly and efficiently.

## How to Use GemLit

1. **Explore**: Navigate through the app using the streamlined sidebar.
2. **Discover**: Access different pages to utilize specific functionalities.
3. **Engage**: Follow user-friendly instructions and input your data or queries.
4. **Experience**: Enjoy the innovative and versatile capabilities of Intellibridge.

## Upcoming Feature: Chat with Data

Currently in development, our **Chat with Data** feature, integrating PandasAI and Langchain, will revolutionize how you interact with data. Automate graph creation and manipulation through simple conversational inputs, making data analysis more intuitive than ever.

    """)

display()