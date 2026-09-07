# ⚽ Football Stats Assistant (RAG Chatbot with LangChain & Groq)

A Retrieval-Augmented Generation (RAG) conversational assistant built with **Streamlit**, **LangChain (LCEL)**, and **Groq Cloud API**. The application processes football player statistics from local PDF documents and answers user queries with domain-specific accuracy, minimizing LLM hallucinations.

---
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://chatbot-rag-for-file-pdf-using-langchain-and-groq-api-js9wvkow.streamlit.app/)

**Live Demo:** [Football Stats Assistant (RAG Chatbot with LangChain & Groq)](https://chatbot-rag-for-file-pdf-using-langchain-and-groq-api-js9wvkow.streamlit.app/)

## 🚀 Key Features

* **Retrieval-Augmented Generation (RAG):** Grounds LLM responses strictly on the provided context (`DB_Football Player Stats.pdf`).
* **Ultra-Fast Inference:** Powered by Groq's LPU acceleration using `ChatGroq`.
* **Modern LCEL Architecture:** Built using LangChain Expression Language (`RunnablePassthrough`, modern prompt formatting, and output parsers) replacing deprecated chain interfaces.
* **Efficient Vector Storage:** Embeds text chunks with HuggingFace's `all-MiniLM-L6-v2` and searches via a local `FAISS` vector database.
* **Cached Pipeline Initialization:** Leverages `@st.cache_resource` to prevent redundant document loading, embedding generation, and FAISS indexing during chat re-renders.

---

## 🛠 Tech Stack

* **Frontend / UI:** Streamlit
* **Orchestration Framework:** LangChain, LangChain Core, LangChain Community
* **LLM Engine:** Groq API (`ChatGroq`)
* **Vector Store & Retrieval:** FAISS (CPU), `sentence-transformers` via `langchain-huggingface`
* **Document Processing:** PyPDF (`PyPDFDirectoryLoader`), `langchain-text-splitters`
* **Environment Management:** `python-dotenv`

---

## 📁 Repository Structure

```text
├── DB/
│   └── DB_Football Player Stats.pdf   # Source document containing football data
├── .gitignore                         # Git exclusion rules (e.g., .env, venv)
├── main.py                            # Streamlit UI & LCEL RAG pipeline
├── requirements.txt                   # Project dependencies
└── README.md                          # Project documentation

```

---

## 🧠 System Workflow

1. **Document Ingestion:** The PDF within the `DB/` directory is extracted via `PyPDFDirectoryLoader`.
2. **Text Chunking:** Text is divided into manageable segments using `RecursiveCharacterTextSplitter` (chunk size: 1000, overlap: 200).
3. **Vector Embeddings & Indexing:** Chunks are embedded using `sentence-transformers/all-MiniLM-L6-v2` and indexed into an in-memory `FAISS` vector index.
4. **Contextual Retrieval:** The retriever queries the top 3 (`k=3`) most relevant passages matching the user's prompt.
5. **LCEL Generation Chain:**

$$\text{Retriever} \longrightarrow \text{Format Context} \longrightarrow \text{Prompt Template} \longrightarrow \text{Groq LLM} \longrightarrow \text{StrOutputParser}$$



---

## ⚙️ Installation & Local Setup

1. **Clone this repository:**
```bash
git clone [https://github.com/katarizkyo99/Chatbot-RAG-for-File-PDF-using-Langchain-and-Groq-API.git](https://github.com/katarizkyo99/Chatbot-RAG-for-File-PDF-using-Langchain-and-Groq-API.git)
cd Chatbot-RAG-for-File-PDF-using-Langchain-and-Groq-API

```


2. **Create and activate a virtual environment:**
```bash
python -m venv venv
source venv/bin/activate    # macOS / Linux
venv\Scripts\activate       # Windows

```


3. **Install dependencies:**
```bash
pip install -r requirements.txt

```


4. **Set up Environment Variables:**
Create a `.env` file in the root directory (do not commit this file):
```env
GROQ_API_KEY=your_groq_api_key_here

```


5. **Run the Streamlit application:**
```bash
streamlit run main.py

```



---

## ☁️ Deployment Configuration (Streamlit Cloud)

When deploying to Streamlit Community Cloud:

1. Push your repository without the `.env` file.
2. In the **Streamlit Cloud Dashboard**, open your application settings:
* Navigate to **Settings** > **Secrets**.
* Add your Groq API key:
```toml
GROQ_API_KEY = "your_actual_groq_api_key"

```




3. Deploy the application.

---

## 👤 Author

* **GitHub:** [@katarizkyo99](https://www.google.com/search?q=https://github.com/katarizkyo99)

```

```
