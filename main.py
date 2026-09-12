import os
import time
import tempfile

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

st.set_page_config(page_title="AskYourPDF", page_icon="📄", layout="centered")

MAX_FILE_SIZE_MB = 2
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

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
def build_retriever(uploaded_files):
    """Build the FAISS retriever from uploaded PDF files.
    Cached by Streamlit based on the uploaded files' content, so re-uploading
    the exact same files won't trigger a rebuild.
    """
    if not uploaded_files:
        return None, "Upload at least one PDF (max 2MB each) in the sidebar to get started."

    with tempfile.TemporaryDirectory() as tmpdir:
        for f in uploaded_files:
            with open(os.path.join(tmpdir, f.name), "wb") as out:
                out.write(f.getbuffer())

        loader = PyPDFDirectoryLoader(tmpdir)
        docs = loader.load()

    if not docs:
        return None, "PDFs were uploaded but no text could be extracted from them."

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
        """You are a helpful assistant that answers questions based only on the provided document context.
If the answer isn't in the context, say that you don't know instead of guessing.

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
    st.header("📄 Documents")
    raw_uploads = st.file_uploader(
        f"Upload PDF files (max {MAX_FILE_SIZE_MB}MB each)",
        type="pdf",
        accept_multiple_files=True,
    )

    valid_files = []
    if raw_uploads:
        for f in raw_uploads:
            if f.size > MAX_FILE_SIZE_BYTES:
                st.error(f"❌ **{f.name}** is {f.size / 1024 / 1024:.2f}MB — exceeds the {MAX_FILE_SIZE_MB}MB limit and was skipped.")
            else:
                valid_files.append(f)
        if valid_files:
            st.success(f"✅ {len(valid_files)} file(s) ready: " + ", ".join(f.name for f in valid_files))

    st.divider()
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

# ---- Main title (custom styled header) ----
st.markdown(
    """
    <style>
    .askyourpdf-header {
        text-align: center;
        padding: 0.5rem 0 1.5rem 0;
    }
    .askyourpdf-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #2563eb, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .askyourpdf-subtitle {
        color: #6b7280;
        font-size: 1.05rem;
    }
    </style>
    <div class="askyourpdf-header">
        <div class="askyourpdf-title">📄 AskYourPDF</div>
        <div class="askyourpdf-subtitle">Upload any PDF, ask anything about it.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---- Build retriever (cached by file content, independent of model choice) ----
retriever, error_msg = build_retriever(tuple(valid_files))

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

if prompt_input := st.chat_input("What would you like to know about this document?"):
    st.chat_message("user").markdown(prompt_input)
    st.session_state.messages.append({"role": "user", "content": prompt_input})

    with st.chat_message("assistant"):
        with st.spinner("Analyzing your document..."):
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
