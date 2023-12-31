import streamlit as st
import textwrap
import base64
from PIL import Image
from io import BytesIO
import google.generativeai as genai
import google.ai.generativelanguage as glm
import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
import textwrap
import chromadb
import numpy as np
import pandas as pd

import google.generativeai as genai
import google.ai.generativelanguage as glm
from chromadb import Documents, EmbeddingFunction, Embeddings


# Load environment variables and configure API
load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

# Helper Functions
def to_markdown(text):
    text = text.replace('•', '  *')
    return textwrap.indent(text, '> ', predicate=lambda _: True)

def process_image(text_for_llm, image_url):
    llm_vision = ChatGoogleGenerativeAI(model="gemini-pro-vision")
    message = HumanMessage(content=[{"type": "text", "text": text_for_llm}, {"type": "image_url", "image_url": image_url}])
    return llm_vision.invoke([message])

def generate_content_based_on_image(user_input, image_bytes):
    model = genai.GenerativeModel('gemini-pro-vision')
    response = model.generate_content(
        glm.Content(
            parts=[
                glm.Part(text=user_input),
                glm.Part(inline_data=glm.Blob(mime_type='image/jpeg', data=image_bytes)),
            ],
        ),
        stream=True
    )
    return response

def display():
    st.title("Story Building with a base Image!")
    uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        image_url = "data:image/jpeg;base64," + base64.b64encode(buffered.getvalue()).decode()

        col1, col2, col3 = st.columns(3)
        if col1.button("What's in the Image?"):
            result = process_image("What do you see in this image?", image_url)
            st.markdown(to_markdown(result.content), unsafe_allow_html=True)

        if col2.button('Generate a Story'):
            result = process_image("Create a short story based on the image", image_url)
            st.markdown(to_markdown(result.content), unsafe_allow_html=True)

        if col3.button('Write about image'):
            result = process_image("Create a short story based on the image", image_url)
            st.markdown(to_markdown(result.content), unsafe_allow_html=True)

display()