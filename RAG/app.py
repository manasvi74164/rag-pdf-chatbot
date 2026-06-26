import streamlit as st
import os
from dotenv import load_dotenv

# Standard LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

st.set_page_config(page_title="RAG PDF Chatbot", page_icon="📚")
st.title("📚 RAG PDF Chatbot")

# --- Authentication Handling ---
env_key = os.getenv("GOOGLE_API_KEY")

st.sidebar.title("Configuration")
if env_key and env_key.startswith("AIzaSy"):
    api_key = env_key
    st.sidebar.success("Valid API Key loaded from environment!")
else:
    api_key = st.sidebar.text_input("Enter Gemini API Key (starts with AIzaSy):", type="password")
    st.sidebar.info("Get a key from [Google AI Studio](https://aistudio.google.com/)")


@st.cache_resource
def process_pdf(uploaded_file):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader("temp.pdf")
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    chunks = splitter.split_documents(documents)

    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-2-preview",
        task_type="RETRIEVAL_DOCUMENT"
    )
    
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    if os.path.exists("temp.pdf"):
        os.remove("temp.pdf")
        
    return vector_db.as_retriever()


# Check if the API key is provided before showing the rest of the app
if not api_key:
    st.warning("Please paste a valid Google AI Studio API Key (starting with `AIzaSy`) in the sidebar to proceed.")
else:
    os.environ["GOOGLE_API_KEY"] = api_key

    file = st.file_uploader("Upload PDF", type="pdf")

    if file:
        with st.spinner("Processing PDF and generating embeddings... Please wait."):
            retriever = process_pdf(file)
        st.success("PDF processed and indexed successfully!")

         
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Answer the question based strictly on the context below.\n\nContext: {context}"),
            ("human", "{input}"),
        ])

        combine_docs_chain = create_stuff_documents_chain(llm, prompt)
        qa_chain = create_retrieval_chain(retriever, combine_docs_chain)

        # Chat interface inside a form to protect API limits
        with st.form("chat_form", clear_on_submit=False):
            question = st.text_input("Ask your question:")
            submit_button = st.form_submit_button("Submit")

        if submit_button and question:
            with st.spinner("Thinking..."):
                try:
                    response = qa_chain.invoke({"input": question})
                    st.write("### Answer:")
                    st.write(response["answer"])
                except Exception as e:
                    st.error(f"An API error occurred: {e}")