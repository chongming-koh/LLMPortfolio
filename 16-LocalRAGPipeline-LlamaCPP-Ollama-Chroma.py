# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 22:25:23 2025

@author: Koh Chong Ming

Local RAG Pipeline (LlamaCPP + Ollama + Chroma)

"""
import os
import glob
import gradio as gr
from tqdm import tqdm 
# imports for langchain and Chroma and plotly

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import LlamaCpp
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

#Chroma storage
db_name = "vector_db"

# ==============================================
# 1. Document Loading (PDF + TXT) with Progress Bar
# ==============================================

knowledge_base_path = "knowledge-base"
folders = glob.glob(os.path.join(knowledge_base_path, "*"))

documents = []

print("Loading documents...")
for folder in tqdm(folders, desc="Loading folders", ncols=100):
    doc_type = os.path.basename(folder)

    # Load PDFs
    pdf_loader = DirectoryLoader(folder, glob="**/*.pdf", loader_cls=PyPDFLoader)
    pdf_docs = pdf_loader.load()

    # Load Text Files
    text_loader = DirectoryLoader(folder, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    text_docs = text_loader.load()

    for doc in pdf_docs + text_docs:
        doc.metadata["doc_type"] = doc_type

    documents.extend(pdf_docs + text_docs)

print(f"Loaded {len(documents)} total documents from {knowledge_base_path}\n")

# ==============================================
# 2. Chunking
# ==============================================

print("Splitting documents into chunks...")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks\n")

doc_types = set(chunk.metadata['doc_type'] for chunk in chunks)
print(f"Document types found: {', '.join(doc_types)}\n")

# ==============================================
# 3. Vectorization with Ollama Embedding Model (Local)
# ==============================================

embedding_model = "embeddinggemma:latest"  # Ollama embedding model


if os.path.exists(db_name):
    print("🧹 Existing Chroma datastore found — deleting old collection...")
    Chroma(persist_directory=db_name, embedding_function=None).delete_collection()
else:
    os.makedirs(db_name, exist_ok=True)

print(f"⚙️ Generating embeddings locally with Ollama model '{embedding_model}'...")
embeddings = OllamaEmbeddings(model=embedding_model)

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"✅ Vectorstore created and saved to '{db_name}' with {vectorstore._collection.count()} vectors")

# ==============================================
# 3a) Load existing Vector data created by embedded LLM previously from Chroma store
# When i do not want to go through the vectorization again.
# ==============================================
'''

embedding_model = "embeddinggemma:latest"  # Ollama embedding model
embeddings = OllamaEmbeddings(model=embedding_model)
vectorstore = Chroma(
    persist_directory=db_name,
    embedding_function=embeddings
)

print("✅ Loaded existing vectorstore!")
print(f"Stored documents: {vectorstore._collection.count()}")

print("✅ Loaded existing vectorstore!")
print(f"Stored documents: {vectorstore._collection.count()}")

# Get one vector and find how many dimensions it has

collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")

'''

# ==============================================
# 4. Retrieval and Local Chat with LlamaCPP
# ==============================================

llama_chat_model_path = r"C:\\LlamaCPP\\models\\Llama-3.2-1B-Instruct-Q8_0.gguf"

llm = LlamaCpp(
    model_path=llama_chat_model_path,
    n_ctx=8192,
    n_batch=256,
    temperature=0.7,
    verbose=False
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

conversation_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# ==============================================
# 5. Gradio Chat UI
# ==============================================

def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    return result["answer"]

view = gr.ChatInterface(chat, type="messages", title="🦙 Local RAG Assistant (Ollama Embedding + LlamaCPP)").launch(inbrowser=True)