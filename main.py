import os
import time

import streamlit as st
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

# ---- Available Groq models (checked against Groq's current production model list) ----
AVAILABLE_MODELS = {
    "GPT-OSS 20B (fastest, default)": "openai/gpt-oss-20b",
    "GPT-OSS 120B (larger, more capable)": "openai/gpt-oss-120b",
    "Llama 3.3 70B Versatile": "llama-3.3-70b-versatile",
    "Llama 3.1 8B Instant": "llama-3.1-8b-instant",
}


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


@st.cache_resource(show_spinner=False)
def build_retriever(db_path: str = "./DB"):
    """Load PDFs, split, embed, and build the FAISS retriever.
    Cached separately from the LLM so switching models does not re-run embeddings.
    """
    if not os.path.isdir(db_path) or not os.listdir(db_path):
        return None, "No PDF files found in the './DB' folder. Add at least one PDF and restart the app."

    loader = PyPDFDirectoryLoader(db_path)
    docs = loader.load()
    if not docs:
        return None, "PDFs were found but no text could be extracted from them."

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs[:20])

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_db = FAISS.from_documents(final_documents, embeddings)
    retriever = vector_db.as_retriever(search_kwargs={"k": 3})
    return retriever, None


def build_rag_chain(retriever, model_id: str, temperature: float):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        st.error("GROQ_API_KEY is missing. Please set it in your environment or .env file.")
        st.stop()

    llm = ChatGroq(groq_api_key=api_key, model_name=model_id, temperature=temperature)

    prompt = ChatPromptTemplate.from_template(
        """You are an expert chatbot in football. Answer the user question based only on the provided context.
If you don't know the answer, just say that you don't know.

Context:
{context}

Question: {question}

Answer:"""
    )

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain


# ---- Sidebar ----
with st.sidebar:
    st.header("⚙️ Settings")

    selected_label = st.selectbox("Model", list(AVAILABLE_MODELS.keys()), index=0)
    selected_model_id = AVAILABLE_MODELS[selected_label]

    temperature = st.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.1,
                             help="Lower = more factual/consistent. Higher = more creative/varied.")

    show_sources = st.toggle("Show retrieved sources", value=True)

    st.divider()
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.caption(f"Active model: `{selected_model_id}`")

# ---- Main title ----
st.title("⚽ Football Stats Assistant")
st.write("Ask questions about football stats based on the documents in your database.")

# ---- Build retriever (cached, independent of model choice) ----
retriever, error_msg = build_retriever()

if error_msg:
    st.warning(error_msg)
    st.stop()

rag_chain = build_rag_chain(retriever, selected_model_id, temperature)

# ---- Chat state ----
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("sources") and show_sources:
            with st.expander("📄 Sources used for this answer"):
                for i, src in enumerate(message["sources"], start=1):
                    st.markdown(f"**Source {i}:**\n\n{src[:500]}{'...' if len(src) > 500 else ''}")

if prompt_input := st.chat_input("Siapa pemain dengan gol terbanyak?"):
    st.chat_message("user").markdown(prompt_input)
    st.session_state.messages.append({"role": "user", "content": prompt_input})

    with st.chat_message("assistant"):
        with st.spinner("Menganalisis statistik..."):
            start = time.time()
            retrieved_docs = retriever.invoke(prompt_input)
            response = rag_chain.invoke(prompt_input)
            elapsed = time.time() - start

        st.markdown(response)
        st.caption(f"⏱️ Answered in {elapsed:.2f}s using {selected_label}")

        source_texts = [doc.page_content for doc in retrieved_docs]
        if show_sources:
            with st.expander("📄 Sources used for this answer"):
                for i, src in enumerate(source_texts, start=1):
                    st.markdown(f"**Source {i}:**\n\n{src[:500]}{'...' if len(src) > 500 else ''}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response,
        "sources": source_texts,
    })
