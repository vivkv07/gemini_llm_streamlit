
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
    page_title="Gemini LLM",
    page_icon="🧊",
    layout="wide",
    initial_sidebar_state="expanded"
)

with st.sidebar:
    st.caption("Developer: Vivek Kv")

    st.markdown('''[![Streamlit App](https://badgen.net/pypi/v/streamlit)](https://pypi.org/project/streamlit/)
                    [![Github Link](https://badgen.net/badge/icon/github?icon=github&label)]()
                    [![BymeaCoffee](https://badgen.net/badge/icon/buymeacoffee?icon=buymeacoffee&label)](https://www.buymeacoffee.com/vivekkovvuru)''')

load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# if __name__ == "__main__":
#     main()
def display():
    st.title("Gemini App - Home")
    
    st.markdown("""
    Welcome to the Gemini App! This app combines various AI models and functionalities to assist you with a range of tasks.
    Here are some of the capabilities of this app:

    - **Ask a Question**: You can ask questions, and the app will provide informative answers.
    
    - **Understand Your Image**: Upload an image, and the app will help you understand its content or even generate a story based on it.
    
    - **Content Generation**: You can provide context or a description, and the app will generate content, such as paragraphs, based on your input.
    
    - **Chat**: Engage in a conversation with a chatbot that can provide information and answer your queries.
    
    - **Document Query**: Upload PDF documents, create vector search tables, and search for relevant passages within those documents.

    **Additional Features**:
    
    - **Vector Search**: Utilize the power of vector search to retrieve relevant information from your documents.
    
    - **Natural Language Processing (NLP)**: The app uses state-of-the-art NLP models to understand and generate text.
    
    - **Image Analysis**: Gain insights from images and create stories using AI-powered image analysis.
    
    - **Content Creation**: Generate content for articles, reports, or any textual content with ease.
    
    - **Interactive Chat**: Engage in a conversation with a friendly chatbot that's here to assist you.

    **How to Use**:
    
    1. Use the sidebar navigation to explore the specific features.
    
    2. Click on the respective pages to access each functionality.
    
    3. Follow the on-screen instructions and input your queries or content.
    
    4. Enjoy the AI-powered capabilities of the Gemini App!
    
    Have fun exploring the app and making the most of its AI features!
    """)

display()