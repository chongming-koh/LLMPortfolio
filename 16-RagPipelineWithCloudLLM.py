# -*- coding: utf-8 -*-
"""
Created on Tue Oct 21 21:39:20 2025

@author: Koh Chong Ming

RAG Pipeline with Chroma + BAAI/bge-multilingual-gemma2 + Meta-Llama-3.1-8B-Instruct

"""
import os
import glob
from openai import OpenAI
import gradio as gr
from tqdm import tqdm 
# imports for langchain and Chroma and plotly

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_nebius import NebiusEmbeddings
from langchain_nebius import ChatNebius

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

key_path = r"C:\PythonStuff\keys\nebius_api_key"
with open(key_path, "r") as file:
    nebius_api_key = file.read().strip()

os.environ["NEBIUS_API_KEY"] = nebius_api_key

# Nebius uses the same OpenAI() class, but with additional details
nebius_client = OpenAI(
    base_url="https://api.studio.nebius.ai/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY"),
)

llama_8b_model = "meta-llama/Meta-Llama-3.1-8B-Instruct"
llama_70b_model ="meta-llama/Llama-3.3-70B-Instruct"
gemma_9b_model = "google/gemma-2-9b-it-fast"
Qwen2_5_72B_model = "Qwen/Qwen2.5-72B-Instruct"
DeepSeek_V33024 ="deepseek-ai/DeepSeek-V3-0324"
openai_20b = "openai/gpt-oss-20b"
Hermes_4_70B_model ="NousResearch/Hermes-4-70B"
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
# 3. Vectorization with BAAI/bge-multilingual-gemma2
# ==============================================

embeddings = NebiusEmbeddings(
    api_key=os.environ["NEBIUS_API_KEY"],
    model="BAAI/bge-multilingual-gemma2"  # or "nebius/BAAI/bge-multilingual-gemma2"
)

# Clean up Chroma datastore

if os.path.exists(db_name):
    Chroma(persist_directory=db_name, embedding_function=embeddings).delete_collection()
    
# Create our Chroma vectorstore!
print("Generating embeddings with bge-multilingual-gemma2 model from Nebius...")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=db_name)
print(f"Vectorstore created with {vectorstore._collection.count()} vectors\n")

# Get one vector and find how many dimensions it has

collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions\n")

# ==============================================
# 3a) Load existing Vector data created by embedded LLM previously from Chroma store
# When i do not want to go through the vectorization again.
# ==============================================
'''
# Instantiate an object named embeddings
embeddings = NebiusEmbeddings(
    api_key=os.environ["NEBIUS_API_KEY"],
    model="BAAI/bge-multilingual-gemma2"  # or "nebius/BAAI/bge-multilingual-gemma2"
)

# Load existing Vector data created by embedded LLM previously from Chroma store
vectorstore = Chroma(
    persist_directory=db_name,
    embedding_function=embeddings
)

print("✅ Loaded existing vectorstore!")
print(f"Stored documents: {vectorstore._collection.count()}")

# Get one vector and find how many dimensions it has

collection = vectorstore._collection
sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
dimensions = len(sample_embedding)
print(f"The vectors have {dimensions:,} dimensions")

'''

# ==============================================
# 4. Retrieval and Chat with Meta-Llama-3.1-8B-Instruct 
# ==============================================

# create a new Chat with Nebius
llm = ChatNebius(
    model=llama_8b_model,
    temperature=0.7,
    api_key=os.getenv("NEBIUS_API_KEY"))

# set up the conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# the retriever is an abstraction over the VectorStore that will be used during RAG
retriever = vectorstore.as_retriever()

# putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# set up a new conversation memory for the chat
memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)

# putting it together: set up the conversation chain with the GPT 4o-mini LLM, the vector store and memory
conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory)

# ==============================================
# 5. Gradio Chat UI
# ==============================================

# Wrapping in a function - note that history isn't used, as the memory is in the conversation_chain

def chat(message, history):
    result = conversation_chain.invoke({"question": message})
    return result["answer"]

# And in Gradio:

view = gr.ChatInterface(chat, type="messages").launch(inbrowser=True)