import streamlit as st
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

def fetch_website_content(url, max_pages=5):
    def get_links(url):
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        return [a['href'] for a in soup.find_all('a', href=True) if a['href'].startswith(url)]
    
    visited = set()
    to_visit = [url]
    content = ""
    
    while to_visit and len(visited) < max_pages:
        current_url = to_visit.pop(0)
        if current_url not in visited:
            try:
                response = requests.get(current_url)
                soup = BeautifulSoup(response.content, 'html.parser')
                content += soup.get_text() + "\n\n"
                visited.add(current_url)
                to_visit.extend([link for link in get_links(current_url) if link not in visited])
            except Exception as e:
                st.error(f"Error fetching {current_url}: {str(e)}")
    
    return content

def create_vector_store(content):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_text(content)
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=GOOGLE_API_KEY
    )
    
    vector_store = FAISS.from_texts(splits, embeddings)
    return vector_store

def answer_question(vector_store, question):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        google_api_key=GOOGLE_API_KEY,
        temperature=0.3
    )
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    
    return qa.run(question)

# Streamlit UI
st.title("Advanced Website Q&A Chatbot (Gemini)")

url = st.text_input("Enter website URL:")
question = st.text_input("Ask a question about the website:")

if url and question:
    with st.spinner("Fetching website content..."):
        content = fetch_website_content(url)
    
    with st.spinner("Creating vector store..."):
        vector_store = create_vector_store(content)
    
    with st.spinner("Analyzing and answering..."):
        answer = answer_question(vector_store, question)
    
    st.write("Answer:", answer)

# Chat History
st.markdown("---")
st.write("Chat History:")

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

for i, (q, a) in enumerate(st.session_state.chat_history, 1):
    st.write(f"Q{i}: {q}")
    st.write(f"A{i}: {a}")

if url and question:
    st.session_state.chat_history.append((question, answer))
