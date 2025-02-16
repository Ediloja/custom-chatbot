import os
import json
from dotenv import load_dotenv
from typing import List
from pinecone import Pinecone

# FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

# RAG
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_pinecone import PineconeVectorStore

# Environment variables
load_dotenv(".env") 
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class Request(BaseModel):
    messages: List[dict]

app = FastAPI(
    title="RAG API",
    description="API para chatbot con RAG utilizando OpenAI y Pinecone",
    version="1.0.3"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)   

# OpenAI
client = OpenAI()

# Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "chatbot"
    index = pc.Index(index_name)
    namespace = "testing-index-local"
    print("Index created successfully!")
except Exception as e:
    print("Error connecting to Pinecone:", e)

# Embeddings
try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
except Exception as e:
    print("Error creating embeddings model:", e)

# Vectorstore y Docstore
vectorstore = PineconeVectorStore(embedding=embeddings, index=index, namespace=namespace)
store = InMemoryStore()

# Splitters para documentos "padre" e "hijo"
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# Definir el ParentDocumentRetriever antes de usarlo en la indexación
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Carga e indexación de documentos (se elimina el contenido actual y se reindexan)
try:
    # Intentar eliminar todos los vectores en el namespace
    try:
        index.delete(delete_all=True, namespace=namespace)
        print(f"Existing vectors in namespace '{namespace}' have been deleted.")
    except Exception as e:
        if "Namespace not found" in str(e):
            print(f"Namespace '{namespace}' not found, skipping deletion.")
        else:
            raise e

    loaders = [
        PyMuPDFLoader("api/assets/calendario-academico-mad-abril-agosto-2025.pdf"),
        PyMuPDFLoader("api/assets/contenido-metacurso.pdf"),
        PyMuPDFLoader("api/assets/plan-docente-mad.pdf"),
        PyMuPDFLoader("api/assets/preguntas-frecuentes-mad.pdf"),
        PyMuPDFLoader("api/assets/preguntas-frecuentes-eva.pdf"),
    ]

    documents = []

    for loader in loaders:
        documents.extend(loader.load())

    print("Documents uploaded successfully!")
    
    # Agregar los documentos al retriever para reindexar
    retriever.add_documents(documents)
except Exception as e:
    print("Error during re-indexing of documents:", e)


def stream_data_with_rag(messages: List[ChatCompletionMessageParam], protocol: str = 'data'):
    # Extraer la última pregunta enviada por el usuario (opcionalmente, para recuperar contexto de documentos)
    last_query = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            last_query = message.get("content", "")
            break
    if not last_query:
        last_query = " "

    # Recuperar documentos relevantes para la última consulta utilizando el retriever.
    docs = retriever.invoke(last_query)
    docs_text = "".join([doc.page_content for doc in docs])

    # Prompt para el chatbot que incluye el contexto recuperado
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise. "
        "Context: {context}"
    )

    system_prompt_formatted = system_prompt.format(context=docs_text)

    # Construir la lista de mensajes que se enviará a OpenAI:
    # Se añade un mensaje del sistema (con el prompt que incluye el contexto) y se concatenan todos los mensajes anteriores.
    new_messages = [{"role": "developer", "content": system_prompt_formatted}] + messages

    if protocol == 'data':
        stream_result = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=new_messages,
            stream=True
        )

        for chunk in stream_result:
            for choice in chunk.choices:
                if choice.finish_reason == "stop":
                    continue
                else:
                    yield '0:{text}\n'.format(text=json.dumps(choice.delta.content))
            # Si el chunk no contiene choices, se envía un mensaje de finalización
            if chunk.choices == []:
                usage = chunk.usage
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                yield 'e:{{"finishReason":"{reason}","usage":{{"promptTokens":{prompt},"completionTokens":{completion}}},"isContinued":false}}\n'.format(
                    reason="stop",
                    prompt=prompt_tokens,
                    completion=completion_tokens
                )

        # Mensaje de cierre del stream
        yield '2:[{"finishReason":"stop","usage":{"promptTokens":0,"completionTokens":0},"isContinued":false}]\n'

# API
@app.post("/api/chat", response_class=StreamingResponse)
async def handle_chat_data(request: Request, protocol: str = Query('data')):
    try:
        messages = request.messages
        response = StreamingResponse(stream_data_with_rag(messages, protocol))
        response.headers["x-vercel-ai-data-stream"] = "v1"
        return response
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "error": "An error occurred during processing the request. Please check the API and try again.",
                "detail": str(e)
            }
        )
    
@app.get("/")
async def root():
    return {"message": "Hello from FastAPI!"}
