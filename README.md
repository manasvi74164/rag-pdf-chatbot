# rag-pdf-chatbot
RAG-based PDF Chatbot built with Streamlit, LangChain, Gemini AI, and FAISS for intelligent question answering from PDF documents.
# RAG PDF Chatbot

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask questions about their content using Google Gemini.

## Features

* Upload PDF documents
* Automatic text extraction
* Document chunking and preprocessing
* Vector embeddings using Gemini Embeddings
* Semantic search with FAISS
* Context-aware question answering
* Streamlit web interface

## Tech Stack

* Python
* Streamlit
* LangChain
* Google Gemini
* FAISS
* PyPDF
* Vector Embeddings

## Workflow

1. Upload a PDF document.
2. Extract text from the PDF.
3. Split content into chunks.
4. Generate embeddings using Gemini.
5. Store embeddings in FAISS.
6. Retrieve relevant chunks based on user queries.
7. Generate answers using Gemini 2.5 Flash.

## Use Cases

* Research Paper Analysis
* Study Material Assistant
* Document Search
* Company Policy Q&A
* Legal Document Exploration
* Knowledge Base Chatbot
