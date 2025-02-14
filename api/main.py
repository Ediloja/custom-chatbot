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
    namespace = "testing-chatbot-1"

except Exception as error:
    print("Error al conectar con Pinecone:", error)

# Embeddings
try:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
except Exception as e:
    print("Error al crear el modelo de embeddings:", e)

# Vectorstore
vectorstore = PineconeVectorStore(embedding=embeddings, index=index, namespace=namespace)
store = InMemoryStore()

# Revisar InMemoryStore
print(store)
print(store.yield_keys)
print(store.yield_keys())

parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

# Loading documents
try:
    loaders = [
        PyMuPDFLoader("api/assets/introduccion-mad.pdf"),
        PyMuPDFLoader("api/assets/calendario-academico-mad-abril-agosto-2025.pdf"),
        PyMuPDFLoader("api/assets/preguntas-frecuentes-mad.pdf"),
        PyMuPDFLoader("api/assets/preguntas-frecuentes-eva.pdf")
        ]

    documents = []

    for loader in loaders:
        documents.extend(loader.load())

    print("Documents uploaded successfully!")
except Exception as error:
    print("Error al cargar los documentos:", error)

# ParentDocumentRetriever
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=store,
    child_splitter=child_splitter,
    parent_splitter=parent_splitter,
    )

retriever.add_documents(documents)


def stream_data_with_rag(messages: List[ChatCompletionMessageParam], protocol: str = 'data'):

    # Extraer la última pregunta enviada por el usuario
    question = ""

    for message in reversed(messages):
        if message.get("role") in ["user", "human", "assistant"]:
            question = message.get("content", "")
            break
    if not question:
        question = " "

    # Recuperar documentos relevantes para la pregunta utilizando el retriever.
    docs = retriever.invoke(question)
    docs_text = "".join([doc.page_content for doc in docs])

    # Prompt para el chatbot
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise. "
        "Context: {context}"
    )

    system_prompt_formatted = system_prompt.format(context=docs_text)

    # Preparar mensajes para la API de OpenAI
    new_messages = [
        {"role": "system", "content": system_prompt_formatted},
        {"role": "user", "content": question}
    ]

    if (protocol == 'data'):
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
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens
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
    