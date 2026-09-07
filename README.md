# Chatbot RAG - Football Player Stats

A lightweight **Retrieval-Augmented Generation (RAG)** chatbot that answers football and player-statistics questions using PDF documents as its knowledge source.

This project demonstrates a simple end-to-end RAG pipeline:

```text
PDF Documents
     │
     ▼
Document Loading
     │
     ▼
Text Chunking
     │
     ▼
Embeddings
     │
     ▼
FAISS Vector Store
     │
     ▼
Retriever
     │
     ▼
Groq LLM
     │
     ▼
Generated Answer
```

The application is implemented as a minimal **Flask web application** with a browser-based chat interface.

---

## Features

* Load multiple PDF documents from the `DB/` directory.
* Split PDF content into smaller text chunks for retrieval.
* Generate document embeddings using Hugging Face Sentence Transformers.
* Store embeddings in an in-memory **FAISS** vector index.
* Retrieve relevant document chunks using semantic similarity.
* Generate responses using **Groq ChatGroq**.
* Provide a simple browser-based chat interface.
* Expose a lightweight REST endpoint for chatbot queries.

---

## System Architecture

The system consists of the following components:

| Component       | Technology         | Description                                                           |
| --------------- | ------------------ | --------------------------------------------------------------------- |
| Frontend        | HTML + JavaScript  | Provides the browser-based chat interface                             |
| Backend         | Flask              | Handles requests, document processing, retrieval, and LLM interaction |
| Document Source | PDF                | Provides the knowledge base                                           |
| Embedding Model | `all-MiniLM-L6-v2` | Converts document chunks into vector representations                  |
| Vector Store    | FAISS              | Performs similarity-based document retrieval                          |
| LLM             | Groq ChatGroq      | Generates answers using retrieved context                             |
| RAG Framework   | LangChain          | Orchestrates document loading, retrieval, and generation              |

### RAG Workflow

When the application starts:

1. PDF documents are loaded from the `DB/` directory.
2. Document content is split into smaller chunks.
3. Each chunk is converted into an embedding.
4. The embeddings are stored in an in-memory FAISS index.
5. The FAISS index is used as the retriever for incoming questions.
6. A user submits a question through the web interface.
7. The relevant document chunks are retrieved.
8. The retrieved context is passed to the Groq LLM.
9. The generated response is returned to the browser.

---

## Tech Stack

### Frontend

* HTML
* JavaScript

### Backend

* Python
* Flask
* python-dotenv

### AI / LLM

* LangChain
* LangChain Groq
* ChatGroq
* Llama3-8b-8192

### Embeddings

* Hugging Face Sentence Transformers
* `sentence-transformers/all-MiniLM-L6-v2`

### Vector Database

* FAISS (`faiss-cpu`)

### Document Processing

* PyPDFDirectoryLoader
* LangChain Community

---

## Project Structure

```text
Chatbot-RAG-File-PDF/
│
├── DB/
│   └── DB_Football Player Stats.pdf
│
├── templates/
│   └── chat.html
│
├── .env
├── .gitignore
├── main.py
└── README.md
```

### Important Files

**`main.py`**

The main application entry point. It is responsible for:

* Loading PDF documents
* Splitting documents into chunks
* Creating embeddings
* Building the FAISS vector store
* Initializing the Groq LLM
* Configuring the RetrievalQA pipeline
* Running the Flask web server
* Handling chatbot requests

**`templates/chat.html`**

Provides the web-based user interface for interacting with the chatbot.

**`DB/`**

Contains the PDF documents used as the chatbot's knowledge base.

**`.env`**

Stores environment variables such as the Groq API key.

---

## Requirements

### Prerequisites

* Python 3.8 or later
* pip
* Internet connection for downloading the embedding model and communicating with the Groq API

### Python Dependencies

Install the required packages with:

```bash
pip install flask python-dotenv langchain langchain-groq langchain-core langchain-community sentence-transformers transformers huggingface-hub faiss-cpu
```

Depending on the environment, `sentence-transformers` may also require additional dependencies such as PyTorch.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/katarizkyo99/Chatbot-RAG-File-PDF.git
cd Chatbot-RAG-File-PDF
```

### 2. Create a Virtual Environment

Creating a virtual environment is recommended.

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install flask python-dotenv langchain langchain-groq langchain-core langchain-community sentence-transformers transformers huggingface-hub faiss-cpu
```

---

## Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

The application requires a valid **Groq API key** to access the configured LLM.

> **Important:** Never commit a real API key, password, or other secret to the repository.

---

## Running the Project

Start the Flask application with:

```bash
python main.py
```

Once the application starts, open:

```text
http://127.0.0.1:5000/
```

The browser-based chatbot interface will then be available.

### Startup Process

At startup, the application:

1. Loads PDF documents from `DB/`.
2. Processes the loaded documents.
3. Splits the documents into chunks.
4. Generates embeddings.
5. Builds an in-memory FAISS index.
6. Initializes the Groq LLM.
7. Starts the Flask server.

The initial indexing process may take some time depending on the size and number of PDF documents.

---

## API Documentation

The application exposes the following endpoints:

| Method | Endpoint | Description                                                  |
| ------ | -------- | ------------------------------------------------------------ |
| `GET`  | `/`      | Serves the chatbot web interface                             |
| `POST` | `/chat`  | Processes a user question and returns the generated response |

### `POST /chat`

#### Request

```json
{
  "message": "Who is the top scorer?"
}
```

#### Response

```json
{
  "response": "Based on the provided documents, ..."
}
```

The actual response depends on the information retrieved from the PDF documents and the generated output from the configured LLM.

### Example Using cURL

```bash
curl -X POST http://127.0.0.1:5000/chat \
-H "Content-Type: application/json" \
-d "{\"message\":\"Who is the top scorer?\"}"
```

---

## Usage

Open the chatbot interface:

```text
http://127.0.0.1:5000/
```

Enter a question related to the football statistics contained in the PDF knowledge base.

For example:

```text
Who is the top scorer?
```

The system retrieves relevant information from the indexed PDF documents and provides an answer generated by the configured Groq LLM.

---

## Knowledge Base

The chatbot uses PDF files located in:

```text
DB/
```

The included knowledge source is:

```text
DB/DB_Football Player Stats.pdf
```

The PDF documents serve as the source of knowledge for the RAG pipeline.

### Vector Index

The FAISS index is generated dynamically when the application starts.

The current implementation:

* Builds the index in memory.
* Does not persist the FAISS index to disk.
* Rebuilds the index every time the application starts.
* Processes the first 20 loaded documents according to the current implementation.

---

## RAG Pipeline

The core Retrieval-Augmented Generation process can be summarized as:

```text
        PDF Documents
              │
              ▼
       Document Loader
              │
              ▼
       Text Splitter
              │
              ▼
      Hugging Face Embeddings
              │
              ▼
        FAISS Vector Store
              │
              ▼
           Retriever
              │
              │ Relevant Context
              ▼
       RetrievalQA Chain
              │
              ▼
         Groq ChatGroq
              │
              ▼
        Generated Answer
```

This approach allows the LLM to generate responses using information retrieved from the project's PDF knowledge base rather than relying solely on the model's internal knowledge.

---

## Troubleshooting

### Missing Groq API Key

If the API key is missing, the application may raise an error indicating that `GROQ_API_KEY` has not been configured.

Make sure `.env` contains:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### FAISS Installation Error

If `faiss-cpu` cannot be installed or imported correctly, install a FAISS version compatible with your operating system and Python environment.

### Slow Startup

The application generates embeddings and creates the FAISS index when it starts. Large or numerous PDF files may increase startup time and memory usage.

### Hugging Face Model Download

The embedding model is downloaded automatically when it is not available locally. An internet connection may therefore be required during the first execution.

### Groq API or Quota Error

Check that:

* The Groq API key is valid.
* The API is accessible.
* The configured model is available.
* The account has sufficient API capacity or quota.

---

## Security

Because the application uses an external LLM API, API credentials must be protected.

### Recommended Practices

* Never commit `.env` files containing real secrets.
* Never expose the Groq API key in frontend code.
* Use environment variables for sensitive configuration.
* Avoid running Flask with `debug=True` in production.
* Use proper secret-management solutions for deployment environments.

---

## Limitations

The current implementation is a minimal RAG prototype and has several limitations:

* The FAISS vector index is rebuilt every time the application starts.
* No persistent vector database is implemented.
* No authentication mechanism is provided.
* API endpoints are not protected by authentication.
* Dependency versions are not pinned.
* Flask debug mode is enabled in the current implementation.
* Large document collections may increase startup time and memory consumption.
* The system depends on the availability of the Groq API.

---

## Development

Possible improvements for future development include:

* Persisting the FAISS index to disk.
* Adding a `requirements.txt` or `pyproject.toml` with pinned versions.
* Implementing configurable chunk size and overlap.
* Improving error handling and logging.
* Adding API authentication.
* Adding conversation history.
* Supporting additional document formats.
* Adding automated tests.
* Deploying the application using Docker.

---

## Docker

The current repository does not include a `Dockerfile` or `docker-compose.yml`.

Containerization can be added in the future to simplify deployment and environment reproducibility.


---

## License

No license file is currently included in the repository.

If this project is intended for public distribution, add an appropriate open-source license such as MIT, Apache 2.0, or another license suitable for the project.

---

## Author

**Rizky Octa Vianto**

GitHub: [@katarizkyo99](https://github.com/katarizkyo99)
