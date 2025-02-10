import os
import json
import threading
import queue
from typing import List

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# IMPORTANTE: Se asume que esta importación corresponde a la versión beta del cliente OpenAI.
from openai import OpenAI, AssistantEventHandler

# Opcional: cargar variables de entorno (por ejemplo, si necesitas claves API)
# from dotenv import load_dotenv
# load_dotenv(".env")

# ---------------------------------------------------
# Configuración de la aplicación FastAPI y CORS
# ---------------------------------------------------
app = FastAPI(
    title="OpenAI Assistant API",
    description="API para chatbot usando el asistente de OpenAI con streaming.",
    version="1.0.0"
)

origins = [
    "http://localhost",
    "http://localhost:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# Modelo de Request: Se espera un objeto JSON con una lista de mensajes
# Cada mensaje debe tener al menos los campos "role" y "content"
# ---------------------------------------------------
class Request(BaseModel):
    messages: List[dict]

# ---------------------------------------------------
# Inicialización global del cliente de OpenAI y creación del asistente
# Se crea una única vez para no recrearlo en cada petición.
# Puedes ajustar el nombre, instrucciones y modelo según tus necesidades.
# ---------------------------------------------------
client = OpenAI()

# ---------------------------------------------------
# Función para generar la respuesta en streaming
# ---------------------------------------------------
def stream_openai_response(messages: List[dict]):
    """
    Esta función:
      1. Crea un hilo (thread) de conversación en OpenAI.
      2. Agrega cada uno de los mensajes recibidos al hilo.
      3. Inicia un run en modo streaming usando un event handler personalizado
         que captura los fragmentos de respuesta.
      4. Transmite cada fragmento al front-end con el formato esperado.
    """

    # 1. Crear un hilo de conversación en OpenAI
    conversation_thread = client.beta.threads.create()
    
    # 2. Agregar los mensajes al hilo (por ejemplo, si tienes mensajes del sistema, usuario y asistente)
    for message in messages:
        client.beta.threads.messages.create(
            thread_id=conversation_thread.id,
            role=message.get("role", "user"),
            content=message.get("content", "")
        )
    
    # 3. Creamos una cola para recibir los chunks de la respuesta
    q = queue.Queue()

    # Definimos un event handler personalizado que extiende AssistantEventHandler.
    # Este handler recibirá cada fragmento (chunk) y lo pondrá en la cola.
    class StreamingEventHandler(AssistantEventHandler):
        def on_text_delta(self, delta, snapshot):
            # delta.value contiene el fragmento de texto
            q.put(delta.value)
        # Puedes sobrescribir otros métodos si necesitas manejar llamadas a herramientas, etc.

    handler = StreamingEventHandler()
    
    # Función que ejecuta el run en modo streaming en un hilo separado para no bloquear el endpoint.
    def run_stream():
        try:
            with client.beta.threads.runs.stream(
                thread_id=conversation_thread.id,
                assistant_id="asst_KDtG6SfNPybQn5XjYEeRF3hx",
                #instructions="Por favor, responde de manera completa y amigable.",
                event_handler=handler,
            ) as stream:
                # Bloquea hasta que el run finalice
                stream.until_done()
        except Exception as e:
            # En caso de error, enviamos un mensaje de error a la cola
            q.put(json.dumps({"error": "Error en el stream", "detail": str(e)}))
        finally:
            # Indicamos el fin del stream colocando un sentinel en la cola
            q.put(None)

    # Iniciar el streaming en un hilo aparte
    stream_thread = threading.Thread(target=run_stream)
    stream_thread.start()

    # 4. Mientras el stream se ejecuta, vamos leyendo la cola y enviando cada fragmento.
    while True:
        chunk = q.get()
        if chunk is None:  # El sentinel indica que el stream terminó
            break
        # Formateamos el chunk de la misma manera que espera el AI SDK de Vercel.
        # Por ejemplo, se utiliza el prefijo "0:" para cada fragmento.
        yield f'0:{json.dumps(chunk)}\n'
    
    # Cuando se termina el stream, enviamos un mensaje final que indique la finalización.
    yield '2:[{"finishReason":"stop","usage":{"promptTokens":0,"completionTokens":0},"isContinued":false}]\n'

# ---------------------------------------------------
# Endpoint de FastAPI para el chat (streaming)
# ---------------------------------------------------
@app.post("/api/chat", response_class=StreamingResponse)
async def handle_chat(request: Request, protocol: str = Query("data")):
    try:
        messages = request.messages
        # Se retorna la StreamingResponse, pasando la función generadora.
        response = StreamingResponse(stream_openai_response(messages), media_type="text/plain")
        response.headers["x-vercel-ai-data-stream"] = "v1"
        return response
    except Exception as e:
        # Manejo genérico de errores
        return JSONResponse(
            status_code=500,
            content={"error": "Error inesperado durante el procesamiento.", "detail": str(e)}
        )