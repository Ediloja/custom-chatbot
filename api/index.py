import os
from dotenv import load_dotenv
from typing import List
import json

#De utils
from .utils.prompt import prompt_template

#FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse

#LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone

# Cargar variables de entorno
load_dotenv(".env.local") 
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

class Request(BaseModel):
    messages: List[dict]

app = FastAPI()

# Modelo de Embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

# Conexión a Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "tutormad" #se está usando un solo index pero con múltiples namespaces
index = pc.Index(index_name) 
namespace = "curso_intro_MAD_2025"
vector_store = PineconeVectorStore(index=index, embedding=embeddings, namespace=namespace)

# Configurar el modelo de Gemini usando Langchain
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=GOOGLE_API_KEY,
    streaming=True,
    temperature=0.3,
)


def stream_rag_response(question: str):
    """Devuelve respuestas generadas con RAG como un stream."""
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 6, "score_threshold": 0.5},
    )

    # Recuperar contexto desde Pinecone
    docs = retriever.invoke(question)
    context = "\n".join(doc.page_content for doc in docs)

    # Construir el mensaje con contexto y pregunta
    input_data = {"context": context, "question": question}

    # Generar la respuesta como streaming desde el LLM
    prompt = ChatPromptTemplate.from_template(prompt_template).format_prompt(**input_data)
    result_stream = llm.stream(prompt.to_messages())

    for chunk in result_stream:
        yield json.dumps({"content": chunk.content}) + "\n"

#API
@app.post("/api/chat")
async def handle_chat_data(question: str, protocol: str = Query('data')):
    try:
        response = StreamingResponse(stream_rag_response(question))
        response.headers["x-vercel-ai-data-stream"] = "v1"
        return response
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
    
@app.get("/")
async def root():
    return {"message": "Hello from FastAPI!"}