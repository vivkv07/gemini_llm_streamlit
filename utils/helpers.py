# Placeholder content for chatbot_helpers.py
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


from chromadb import Documents, EmbeddingFunction, Embeddings

# Load environment variables and configure API
load_dotenv()
genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))

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

class GeminiEmbeddingFunction(EmbeddingFunction):
  def __call__(self, input: Documents) -> Embeddings:
    model = 'models/embedding-001'
    title = "Custom query"
    return genai.embed_content(model=model,
                                content=input,
                                task_type="retrieval_document",
                                title=title)["embedding"]
