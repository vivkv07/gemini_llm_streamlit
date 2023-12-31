# Placeholder content for document_query.py
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
from chromadb.config import Settings

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

# for m in genai.list_models():
#   if 'embedContent' in m.supported_generation_methods:
#     print(m.name)

class GeminiEmbeddingFunction(EmbeddingFunction):
  def __call__(self, input: Documents) -> Embeddings:
    model = 'models/embedding-001'
    title = "Custom query"
    return genai.embed_content(model=model,
                                content=input,
                                task_type="retrieval_document",
                                title=title)["embedding"]

import os
import PyPDF2
import chromadb
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings

# Function to convert PDF to text
# def pdf_to_text(file_path):
#     pdf_file = open(file_path, 'rb')
#     pdf_reader = PyPDF2.PdfReader(pdf_file)
#     text = ""
#     for page_num in range( len(pdf_reader.pages)):
#         text += pdf_reader.pages[page_num].extract_text()
#     pdf_file.close()
#     return text
from io import BytesIO

# Function to convert PDF to text
def pdf_to_text(uploaded_file):
    pdf_file = BytesIO(uploaded_file.read())  # Read the content of the uploaded file
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        text += pdf_reader.pages[page_num].extract_text()
    pdf_file.close()
    return text

def get_relevant_passage(query, db):
  passage = db.query(query_texts=[query], n_results=1)['documents'][0][0]
  return passage

def make_prompt(query, relevant_passage):
  escaped = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
  print(escaped)
  prompt = ("""You are a helpful and informative bot that answers questions using text from the reference passage included below. \
  Be sure to respond in a complete sentence, being comprehensive, including all relevant background information. \
  However, you are talking to a technical audience, so be sure to break down complicated concepts and \
  strike a friendly and converstional tone. Elaborate your answers ocassionally.\
  If the passage is irrelevant to the answer, you may ignore it.
  QUESTION: '{query}'
  PASSAGE: '{relevant_passage}'

    ANSWER:
  """).format(query=query, relevant_passage=escaped)

  return prompt


def display():
    st.title("Work with your Documents")

    # Allow users to choose between existing databases or creating a new one
    create_new_db = st.selectbox('Pick an Option',('Create a new vector search database', 'Use Sample'))


    # Initialize Chroma DB client
    chroma_client = chromadb.Client()
 
    if create_new_db and create_new_db == "Create a new vector search database":
        # Input field for table name
        table_name = st.text_input("Enter a name for the vector search table:")
       
        uploaded_file = st.file_uploader("Upload a PDF document (Limit: 1 file)", type=['pdf'], accept_multiple_files=False)
        
        # if st.button("create"):
        if table_name and uploaded_file and st.button("create"):
            with st.status("processing"):
                # Convert PDF to text
                text = pdf_to_text(uploaded_file)
                st.info("conversion to text is complete!")
                # Initialize text splitter and embeddings
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
                # Split text into chunks
                chunks = text_splitter.split_text(text)
                st.info("Text Chunks are Ready!")
                # Convert chunks to vector representations
                vectors = []
                model = 'models/embedding-001'
                for i, chunk in enumerate(chunks):
                    title = uploaded_file.name
                    vector = genai.embed_content(
                        model=model,
                        content=chunk,
                        task_type="retrieval_document",
                        title=title
                    )["embedding"]
                    vectors.append(vector)

                # Create a new vector search table with the specified name
                chroma_client = chromadb.PersistentClient(path="./chromadb/", settings=Settings(allow_reset=True))

                db = chroma_client.get_or_create_collection(name=table_name, embedding_function=GeminiEmbeddingFunction())

                # Store vectors in the new table
                db.add(
                    embeddings=vectors,
                    documents=chunks,
                    ids=[f"{uploaded_file.name}_{i}" for i in range(len(chunks))]
                )
                st.success(f"Document uploaded, vector representations created, and table '{table_name}' created.")

    else:
        # Allow users to select from existing vector search databases
        chroma_client = chromadb.PersistentClient(path="./chromadb/", settings=Settings(allow_reset=True))
        collections = chroma_client.list_collections()
        
        # st.write(collections)
        selected_db = st.selectbox("Select an existing vector search database:", [collection.name for collection in collections])

        # Allow users to input a query
        query = st.text_input("Enter your query:")
        if st.button("Search"):
            # Perform a query and display the relevant passage
            selected_collection = [collection for collection in collections if collection.name == selected_db][0]
            db = chroma_client.get_or_create_collection(name=selected_db, embedding_function=GeminiEmbeddingFunction())
            relevant_passage = get_relevant_passage(query, db)

            # st.write("Relevant Passage:")
            # st.write(relevant_passage)
            
            prompt = make_prompt(query, relevant_passage)
            # st.write(prompt)

            model = genai.GenerativeModel('gemini-pro')
            response = model.generate_content(prompt)
            st.markdown(response.text)

display()