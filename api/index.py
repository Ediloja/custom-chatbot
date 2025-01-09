import os
from dotenv import load_dotenv
from typing import List

#De utils
from utils.prompt import prompt_template

#FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse

#LangChain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
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

def rag(pregunta):
  retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"k": 6, "score_threshold": 0.5},
  )

  chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
  )
  result = chain.invoke(pregunta)
  print(result)


#API
@app.post("/api/chat")
async def handle_chat_data(request: Request, protocol: str = Query('data')):
    try:
        messages = request.messages
        #response = StreamingResponse(stream_text(messages))
        #response.headers['x-vercel-ai-data-stream'] = 'v1'
        #return response
        return {"message": "Hello from FastAPI!"}
    except Exception as e:
        import traceback
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }
    
@app.get("/")
async def root():
    return {"message": "Hello from FastAPI!"}