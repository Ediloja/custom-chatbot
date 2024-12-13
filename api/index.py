import os
import json
from typing import List
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse

# Importaciones de Langchain
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# De utils
from utils.prompt import prompt_general

# Cargar variables de entorno
load_dotenv(".env.local")

app = FastAPI()

# Configurar el modelo de Gemini usando Langchain
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.environ.get("GEMINI_API_KEY"),
    streaming=True,
    temperature=0.3
)

class Request(BaseModel):
    messages: List[dict]

def convert_messages_to_langchain(messages: List[dict]):
    """Convierte los mensajes al formato de Langchain."""
    langchain_messages = [
        SystemMessage(content=prompt_general)
    ]
    
    for msg in messages:
        role = msg.get('role')
        content = msg.get('content')
        if role == 'user':
            langchain_messages.append(HumanMessage(content=content))
        elif role == 'assistant':
            langchain_messages.append(AIMessage(content=content))
    
    return langchain_messages

def stream_text(messages: List[dict]):
    try:
        # Convertir mensajes al formato de Langchain
        chat_history = convert_messages_to_langchain(messages)
        
        # Crear un template de prompt con el historial de chat
        prompt = ChatPromptTemplate.from_messages(chat_history)
        
        # Crear cadena con output parser para streaming
        chain = prompt | llm | StrOutputParser()
        
        # Generar stream de respuesta
        for chunk in chain.stream({}):
            # Formato similar al stream de OpenAI
            yield '0:{text}\n'.format(text=json.dumps(chunk))

        # Chunk final de finalización
        yield 'e:{{"finishReason":"stop","usage":{{"promptTokens":0,"completionTokens":0}},"isContinued":false}}\n'

    except Exception as e:
        import traceback
        # Imprimir error detallado
        print(f"Error en stream_text: {str(e)}")
        print(traceback.format_exc())
        
        # Enviar mensaje de error
        yield '0:{text}\n'.format(text=json.dumps(f"Error interno: {str(e)}"))

@app.post("/api/chat")
async def handle_chat_data(request: Request, protocol: str = Query('data')):
    try:
        messages = request.messages
        response = StreamingResponse(stream_text(messages))
        response.headers['x-vercel-ai-data-stream'] = 'v1'
        return response
    except Exception as e:
        import traceback
        # Imprimir error completo para depuración
        print(f"Error en handle_chat_data: {str(e)}")
        print(traceback.format_exc())
        
        # Devolver un error más informativo
        return {
            "error": str(e),
            "traceback": traceback.format_exc()
        }