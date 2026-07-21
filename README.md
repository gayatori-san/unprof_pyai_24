# 📚PDF RAG API with FastAPI, LangChain & FAISS

# 🚀 Features

* Upload PDF documents
* Automatic text extraction
* Recursive document chunking
* OpenAI Embeddings
* Local FAISS vector database
* Retrieval-Augmented Generation (RAG)
* FastAPI REST API
* Interactive Swagger documentation
* Concise AI-generated answers
* Temporary file cleanup after processing

---

# 🏗️ Project Structure

```text
.
├── task 24.py
├── README.md
├── requirements.txt
└── uploads/        (optional)
```

---

# ⚙️ Architecture

```text
                 Upload PDF
                      │
                      ▼
              FastAPI /upload
                      │
                      ▼
              PyPDFLoader
                      │
                      ▼
      RecursiveCharacterTextSplitter
                      │
                      ▼
           OpenAI Embeddings
                      │
                      ▼
            FAISS Vector Store
                      │
                      ▼
          Retrieval Chain (LangChain)
                      │
                      ▼
              FastAPI /ask
                      │
                      ▼
          Relevant Document Chunks
                      │
                      ▼
              OpenAI Chat Model
                      │
                      ▼
                Final Answer
```

---

# 🔄 Workflow

## Step 1 — Upload a PDF

The user uploads a PDF document using the `/upload` endpoint.

Example:

```http
POST /upload
```

The uploaded file is temporarily stored before processing.

---

## Step 2 — Extract Text

The application loads the PDF using LangChain's **PyPDFLoader**.

```python
loader = PyPDFLoader(temp_file_path)
docs = loader.load()
```

---

## Step 3 — Split the Document

Large documents are divided into smaller overlapping chunks.

Configuration:

```python
chunk_size = 1000
chunk_overlap = 200
```

This improves retrieval accuracy.

---

## Step 4 — Generate Embeddings

Each chunk is converted into a semantic vector using OpenAI Embeddings.

```python
embeddings = OpenAIEmbeddings()
```

---

## Step 5 — Store Vectors

The vectors are indexed in a local FAISS vector database.

```python
vector_store = FAISS.from_documents(
    splits,
    embeddings
)
```

The retriever returns the three most relevant chunks.

```python
search_kwargs={"k": 3}
```

---

## Step 6 — Build the RAG Pipeline

The application creates a Retrieval Chain consisting of:

* Retriever
* Prompt Template
* ChatOpenAI
* Stuff Documents Chain

This enables context-aware question answering.

---

## Step 7 — Ask Questions

Users send questions to the `/ask` endpoint.

The retriever searches the vector database and provides relevant context to the language model.

The model generates a grounded response based only on the uploaded document.

---

# 📦 Installation

## Clone the repository

```bash
git clone https://github.com/gayatori-san/unprof_pyai_24

cd pdf-rag-api
```

---

## Create a Virtual Environment

### Windows

```bash
python -m venv venv

venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv venv

source venv/bin/activate
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

or

```bash
pip install fastapi
pip install uvicorn
pip install langchain
pip install langchain-openai
pip install langchain-community
pip install langchain-text-splitters
pip install faiss-cpu
pip install pypdf
pip install python-multipart
pip install openai
pip install tiktoken
```

---

# 🔑 Configure API Key

Linux/macOS

```bash
export OPENAI_API_KEY="your_api_key"
```

Windows CMD

```cmd
set OPENAI_API_KEY=your_api_key
```

Windows PowerShell

```powershell
$env:OPENAI_API_KEY="your_api_key"
```

---

# ▶️ Running the Application

Start the API server.

```bash
python main.py
```

or

```bash
uvicorn main:app --reload
```

The application runs at:

```text
http://127.0.0.1:8000
```

---

# 📚 Interactive Documentation

FastAPI automatically provides API documentation.

### Swagger UI

```text
http://127.0.0.1:8000/docs
```

### ReDoc

```text
http://127.0.0.1:8000/redoc
```

---

# 🌐 API Endpoints

## POST `/upload`

Uploads and indexes a PDF.

### Request

Multipart form-data

```
file = sample.pdf
```

### Response

```json
{
  "message": "Successfully processed and indexed 'sample.pdf'. You can now hit the /ask endpoint!"
}
```

---

## POST `/ask`

Ask questions about the uploaded document.

### Request

```json
{
  "question": "What is Retrieval-Augmented Generation?"
}
```

### Response

```json
{
  "answer": "Retrieval-Augmented Generation combines document retrieval with language models to generate context-aware responses."
}
```

---

# 🧠 Technologies Used

* Python 3
* FastAPI
* LangChain
* OpenAI GPT-3.5 Turbo
* OpenAI Embeddings
* FAISS
* PyPDF
* Pydantic
* Uvicorn

---

# 📄 Requirements

Example `requirements.txt`

```text
fastapi
uvicorn
langchain
langchain-openai
langchain-community
langchain-text-splitters
faiss-cpu
pypdf
python-multipart
openai
tiktoken
```

---

# 🔮 Future Improvements

Potential enhancements include:

* Support multiple PDF uploads
* Persistent FAISS index
* ChromaDB or Pinecone integration
* Conversation memory
* Source document citations
* Streaming AI responses
* User authentication
* Docker deployment
* OCR support for scanned PDFs
* Cloud storage integration
* Web interface using Streamlit or React

---

# 🎯 Learning Outcomes

After completing this project, you will understand how to:

* Build a document-based RAG system
* Create REST APIs using FastAPI
* Upload files through HTTP endpoints
* Extract text from PDFs
* Split documents into semantic chunks
* Generate OpenAI embeddings
* Store vectors using FAISS
* Build LangChain retrieval pipelines
* Answer questions grounded in uploaded documents
* Deploy AI-powered backend services

