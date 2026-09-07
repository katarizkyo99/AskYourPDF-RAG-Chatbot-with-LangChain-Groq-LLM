import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

st.set_page_config(page_title="Football Stats Bot", page_icon="⚽", layout="centered")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

@st.cache_resource
def initialize_rag_pipeline():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY is missing. Please set it in your environment.")
        st.stop()
        
    llm = ChatGroq(groq_api_key=api_key, model_name="openai/gpt-oss-20b")
    
    loader = PyPDFDirectoryLoader("./DB")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:20])
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(final_documents, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    
    prompt = ChatPromptTemplate.from_template(
        """You are an expert chatbot in football. Answer the user question based only on the provided context.
If you don't know the answer, just say that you don't know.

Context:
{context}

Question: {question}

Answer:"""
    )
    
    # Modern LCEL Chain (menggantikan RetrievalQA)
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

rag_chain = initialize_rag_pipeline()

st.title("⚽ Football Stats Assistant")
st.write("Tanyakan statistik pemain sepak bola berdasarkan dokumen database!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt_input := st.chat_input("Siapa pemain dengan gol terbanyak?"):
    st.chat_message("user").markdown(prompt_input)
    st.session_state.messages.append({"role": "user", "content": prompt_input})

    with st.spinner("Menganalisis statistik..."):
        # LCEL menggunakan .invoke(), bukan .run()
        response = rag_chain.invoke(prompt_input)
        
    with st.chat_message("assistant"):
        st.markdown(response)
        
    st.session_state.messages.append({"role": "assistant", "content": response})