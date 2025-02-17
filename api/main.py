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

# Current date with timezone (using zoneinfo, available in Python 3.9+)
from zoneinfo import ZoneInfo
import datetime

# Environment variables
load_dotenv(".env") 
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

class Request(BaseModel):
    messages: List[dict]

app = FastAPI(
    title="RAG API",
    description="API for chatbot using RAG with OpenAI and Pinecone",
    version="1.0.3"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)   

# OpenAI client
client = OpenAI()

# Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "chatbot"
    index = pc.Index(index_name)
    namespace = "testing-chatbot-local"
    print("Index created successfully!")
except Exception as exc:
    print("Error connecting to Pinecone:", exc)

# Embeddings
try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    print("Embeddings model created successfully!")
except Exception as exc:
    print("Error creating embeddings model:", exc)

# Vectorstore and Docstore
vectorstore = PineconeVectorStore(embedding=embeddings, index=index, namespace=namespace)
store = InMemoryStore()

# Splitters for parent and child documents
parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# Create the ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
)

# Load and index documents (delete existing vectors and re-index)
try:
    try:
        index.delete(delete_all=True, namespace=namespace)
        print(f"Existing vectors in namespace '{namespace}' have been deleted.")
    except Exception as exc:
        if "Namespace not found" in str(exc):
            print(f"Namespace '{namespace}' not found, skipping deletion.")
        else:
            raise exc

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
    retriever.add_documents(documents)
except Exception as exc:
    print("Error during re-indexing of documents:", exc)


def truncate_messages(messages: List[dict], max_tokens: int = 1500) -> List[dict]:
    """
    Función auxiliar que trunca el historial de mensajes para no exceder un número máximo de tokens.
    Esta implementación usa una heurística simple: mantiene los mensajes más recientes hasta llegar al límite.
    Para un conteo más exacto, se puede usar una biblioteca como 'tiktoken'.
    """
    # En esta versión simple, conservamos los últimos N mensajes (por ejemplo, los 6 últimos)
    return messages[-6:]

def stream_data_with_rag(messages: List[ChatCompletionMessageParam], protocol: str = 'data'):
    current_date = datetime.datetime.now(ZoneInfo("America/Guayaquil")).strftime("%A, %Y-%m-%d")
    
    # Extract the last user query for context retrieval
    last_query = ""

    for message in reversed(messages):
        if message.get("role") == "user":
            last_query = message.get("content", "")
            break
            
    if not last_query:
        last_query = " "

    # Retrieve documents relevant to the last query
    docs = retriever.invoke(last_query)
    docs_text = "".join([doc.page_content for doc in docs])

    # Build the system prompt
    system_prompt = (
        f"You are an assistant for question-answering tasks. Today is {current_date}. "
        "Use the following pieces of retrieved context to help answer the question. "
        "If the context is not sufficient, feel free to use your general knowledge. "
        "Provide your answer in up to three concise sentences. "
        "Context: {context}"
    )

    system_prompt_formatted = system_prompt.format(context=docs_text)

    # Truncate the conversation history to optimize token usage
    truncated_messages = truncate_messages(messages, max_tokens=1500)
    
    # Construct the new messages to send to OpenAI by prepending the system prompt
    new_messages = [{"role": "system", "content": system_prompt_formatted}] + truncated_messages

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
            if chunk.choices == []:
                usage = chunk.usage
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                yield 'e:{{"finishReason":"{reason}","usage":{{"promptTokens":{prompt},"completionTokens":{completion}}},"isContinued":false}}\n'.format(
                    reason="stop",
                    prompt=prompt_tokens,
                    completion=completion_tokens
                )

        yield '2:[{"finishReason":"stop","usage":{"promptTokens":0,"completionTokens":0},"isContinued":false}]\n'

# API endpoint to handle chat requests
@app.post("/api/chat", response_class=StreamingResponse)
async def handle_chat_data(request: Request, protocol: str = Query('data')):
    try:
        messages = request.messages
        response = StreamingResponse(stream_data_with_rag(messages, protocol))
        response.headers["x-vercel-ai-data-stream"] = "v1"
        return response
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content={
                "error": "An error occurred during processing the request. Please check the API and try again.",
                "detail": str(exc)
            }
        )
    
@app.get("/")
async def root():
    return {"message": "Hello from FastAPI!"}