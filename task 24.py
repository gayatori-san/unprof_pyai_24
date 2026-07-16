import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import uvicorn

# LangChain imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

app = FastAPI(
    title="PDF RAG API",
    description="Upload a PDF and ask questions about it."
)

# Global variables to hold our vector store and RAG chain in memory
# Note: In a production app, you would use a persistent database (like Postgres/pgvector or Pinecone)
vector_store = None
rag_chain = None

# Pydantic Model for the Question Request
class QueryRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """
    Accepts a PDF, chunks it, creates embeddings, and indexes them in FAISS.
    """
    global vector_store, rag_chain
    
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed.")
        
    # 1. Save the uploaded file temporarily to disk
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 2. Load and Split the PDF
        loader = PyPDFLoader(temp_file_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # 3. Create Embeddings and Store Vectors
        embeddings = OpenAIEmbeddings()
        vector_store = FAISS.from_documents(splits, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # 4. Initialize the RAG Chain
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, just say that you don't know. "
            "Keep the answer concise.\n\n"
            "Context: {context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return {"message": f"Successfully processed and indexed '{file.filename}'. You can now hit the /ask endpoint!"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
        
    finally:
        # 5. Clean up the temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/ask")
async def ask_question(request: QueryRequest):
    """
    Takes a user query, retrieves relevant chunks from the PDF, and returns an AI answer.
    """
    global rag_chain
    
    if rag_chain is None:
        raise HTTPException(status_code=400, detail="No PDF has been uploaded and indexed yet. Please hit /upload first.")
        
    # Send the question to the LangChain RAG pipeline
    response = rag_chain.invoke({"input": request.question})
    
    return {"answer": response["answer"]}

if __name__ == "__main__":
    print("🚀 Starting PDF RAG API on http://127.0.0.1:8000")
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)