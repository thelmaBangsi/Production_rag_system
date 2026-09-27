# app.py
import db_patch  # Must remain line 1
import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from langchain_chroma import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()
os.environ["ANONYMIZED_TELEMETRY"] = "False"

app = FastAPI(title="MINAGRI Policy Assistant API")

api_key = os.getenv("GROQ_API_KEY")

def get_best_available_model(key):
    url = "https://api.groq.com/openai/v1/models"
    headers = {"Authorization": f"Bearer {key}"}
    preferred_models = [
        "llama-3.3-70b-versatile",
        "llama-3.1-8b-instant",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
        "llama3-8b-8192",
        "mixtral-8x7b-32768"
    ]
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json().get("data", [])
            active_ids = {m["id"] for m in data}
            for pref in preferred_models:
                if pref in active_ids:
                    return pref
            text_models = [m_id for m_id in active_ids if "whisper" not in m_id and "safeguard" not in m_id]
            if text_models:
                return text_models[0]
    except Exception:
        pass
    return "llama3-8b-8192"

selected_model = get_best_available_model(api_key)
embeddings = FastEmbedEmbeddings()
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

llm = ChatGroq(
    model_name=selected_model,
    temperature=0.9,
    groq_api_key=api_key
)

template = """You are an AI assistant specializing in Rwandan agricultural policy (MINAGRI).
Answer the user's question accurately using ONLY the context provided below. 
If the answer is not explicitly mentioned in the context, state clearly that the information is not available in the ingested documents.

Context:
{context}

Question: {question}

Answer (Include document references where applicable):"""

prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("filename", "Unknown Document")
        formatted.append(f"[Source: {source}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

class QueryRequest(BaseModel):
    query: str

@app.post("/api/v1/query")
async def query_rag(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty")
    
    def generate():
        for chunk in rag_chain.stream(request.query):
            yield chunk

    return StreamingResponse(generate(), media_type="text/plain")