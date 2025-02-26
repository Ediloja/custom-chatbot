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
from pymupdf4llm import to_markdown
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_pinecone import PineconeVectorStore

# Environment variables
load_dotenv(".env")
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Define request schema for FastAPI
class Request(BaseModel):
    messages: List[dict]

app = FastAPI(
    title="RAG API",
    description="API for chatbot using RAG with OpenAI and Pinecone",
    version="1.0.3"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://custom-chatbot-front-end.vercel.app"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI
client = OpenAI()

# Global variables for heavy initialization
pc = None
index = None
namespace = None
embeddings = None
vectorstore = None
store = None
parent_splitter = None
child_splitter = None
retriever = None

# Startup event
@app.on_event("startup")
async def startup_event():
    global pc, index, namespace, embeddings, vectorstore, store, parent_splitter, child_splitter, retriever
    # Initialize Pinecone connection and index
    try:
        pc = Pinecone(api_key=PINECONE_API_KEY)
        index_name = "chatbot"
        index = pc.Index(index_name)
        namespace = "chatbot-local"
        print("Index connected successfully!")
    except Exception as exc:
        print("Error connecting to Pinecone:", exc)

    # Create embeddings model
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        print("Embeddings model created successfully!")
    except Exception as exc:
        print("Error creating embeddings model:", exc)

    # Initialize vectorstore and in-memory docstore
    vectorstore = PineconeVectorStore(embedding=embeddings, index=index, namespace=namespace)
    store = InMemoryStore()

    # Create text splitters for parent and child document chunks
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=2000)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)

    # Create the document retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
    )

    # Load and index documents
    try:
        # Delete existing vectors in the namespace
        try:
            index.delete(delete_all=True, namespace=namespace)
            print(f"Existing vectors in namespace '{namespace}' have been deleted.")
        except Exception as exc:
            if "Namespace not found" in str(exc):
                print(f"Namespace '{namespace}' not found, skipping deletion.")
            else:
                raise exc

        docs_markdown = []

        for doc_path in [
            "api/assets/calendario-academico-mad-abril-agosto-2025.pdf",
            "api/assets/guia-didactica-mad.pdf",
            "api/assets/preguntas-frecuentes-mad.pdf",
            "api/assets/preguntas-frecuentes-eva.pdf",
            "api/assets/plan-docente-modificado.pdf"
        ]:
            # Convert each PDF into markdown pages (split by page)
            markdown_pages = to_markdown(doc=doc_path, page_chunks=True)
            for page_data in markdown_pages:
                filtered_metadata = {
                    "source": os.path.basename(doc_path),
                    "page": page_data["metadata"]["page"],
                    "page_count": page_data["metadata"]["page_count"]
                }
                docs_markdown.append({
                    "metadata": filtered_metadata,
                    "text": page_data["text"]
                })

        # Convert markdown data into LangChain Document objects
        docs = [
            Document(
                page_content=doc["text"],
                metadata=doc["metadata"]
            )
            for doc in docs_markdown
        ]

        # Add documents to the retriever
        retriever.add_documents(docs)
        print(f"Inserted {len(docs)} documents in index '{index_name}' and namespace '{namespace}'.")
    except Exception as exc:
        print("Error during re-indexing of documents:", exc)

# Helper function to truncate conversation history to avoid exceeding token limits
def truncate_messages(messages: List[dict], max_tokens: int = 1500) -> List[dict]:
    # Simple heuristic: retain the last 6 messages
    return messages[-6:]

# Main function to process chat queries using RAG
def stream_data_with_rag(messages: List[ChatCompletionMessageParam], protocol: str = 'data'):
    # Extract the last user query for context retrieval
    last_query = ""
    for message in reversed(messages):
        if message.get("role") == "user":
            last_query = message.get("content", "")
            break
    if not last_query:
        last_query = " "

    # Retrieve documents relevant to the user's last query 
    docs = retriever.invoke(last_query)
    context_items = [
            #'Documento: "guia-didactica-mad.pdf" - Contenido: "..." '
            f"Documento: \"{doc.metadata.get('source', 'Desconocido')}\" - Cotenido: \"{doc.page_content}:\" "
            for doc in docs   
        ]
    #Formatear como una sola cadena de texto
    docs_text = "\n".join(context_items)

    # Build the system prompt to enforce that answers are based primarily on the document context.
    system_prompt = (
    """
    # Instrucciones para el Sistema:
        Genera respuestas para las preguntas del usuario únicamente a partir del contexto proporcionado.
        FINGE que la información proporcionada en 'CONTEXTO' es de tu conocimiento general para que la interacción sea más agradable.
        EVITA FRASES como 'según la información', 'según los documentos' 'de acuerdo a la información' etc.
        Responde con explicaciones claras y detalladas. 
        Asegúrante de proporcionar los enlaces que vienen dentro del contexto proporcionalo, como recomendación para el usuario y su aprendizaje;
        Si la pregunta está fuera de contexto no la respondas y menciona que solo posees información del curso de introducción.
        A las palabras más importantes de tu respuesta resaltalas con negrita
        Si la respuesta implica pasos a seguir, enuméralos en una lista clara usando: 1. 2. 3. o con viñetas para mayor claridad.
        - Cuando el usuario pregunte sobre actividades del curso, prioriza la información del documento de plan-docente. 
        - Solo menciona actividades de la guia-didactica si no hay detalles específicos en el documento de plan-docente.
    ### Explicación de los documentos:
        Documento Plan Docente (plan-docente-modificado.pdf): “Este documento contiene las actividades específicas de este curso de introducción a la modalidad a distancia. Usa esta información para responder preguntas sobre qué actividades realizar.”
        Documento Guíá Didáctica (guia-didactica-mad.pdf): “Este documento explica las actividades típicas que se suelen encontrar en cursos a distancia, además de información complementaria para el plan docente. Solo úsalo si no hay información relevante en el documento plan docente.”
    ## Manejo de referencias:
    - El usuario quiere saber de que documentos se extrae la información. Incluye la referencia al final de tu respuesta.
    - Utiliza la siguiente estructura para referenciar los documentos:
    *Extraído de *[fuente 1]* y *[fuente 2]**.
    Por ejemplo: “Extraído del Plan Docente y la Guía Didáctica” o “Extraído de la Guía Didáctica y [FAQ Estudiantes](https://utpl.instructure.com/courses/25885/pages/faq-estudiantes)”.
    - Para idenfiticar el nombre de las fuentes sigue estas instrucciones específicas:
        - Si el documento es: **(preguntas-frecuentes-eva.pdf, preguntas-frecuentes-mad.pdf):** No menciones el nombre del documento; en su lugar di que la información proviene de: [FAQ Estudiantes](https://utpl.instructure.com/courses/25885/pages/faq-estudiantes) al final de tu respuesta.
        - Si el documento es: **plan-docente-modificado.pdf:** Menciona que la información proviene "del Plan Docente".
        - Si el documento es: **guia-didactica-mad.pdf:** Menciona que la información proviene "de la Guía Didáctica".
        - Si el documento es: **calendario-academico-mad-abril-agosto-2025.pdf:** Menciona que la información proviene "del Calendario Académico".
    Aunque sean múltiples documentos en el contexto, menciona exclusivamente los nombres de los documentos que se han utilizado en la respuesta.
    # Contexto: 
    {context}
    """
    )


    system_prompt_formatted = system_prompt.format(context=docs_text)

    # Truncate the conversation history to optimize token usage
    truncated_messages = truncate_messages(messages, max_tokens=1500)
    
    # Construct new messages by prepending the system prompt to the conversation history
    new_messages = [{"role": "system", "content": system_prompt_formatted}] + truncated_messages

    # Call OpenAI API with deterministic settings for consistency
    if protocol == 'data':
        stream_result = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=new_messages,
            temperature=0,
            top_p=1,
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
    
# Root endpoint for quick health check
@app.get("/")
async def root():
    return {"message": "Hello from FastAPI!"}
