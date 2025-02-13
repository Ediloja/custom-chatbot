import os
from dotenv import load_dotenv
from typing import List
import json

# De utils
from .utils.prompt import prompt_template

# FastAPI
from pydantic import BaseModel
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI
from openai.types.chat.chat_completion_message_param import ChatCompletionMessageParam

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

client = OpenAI()

def stream_data(messages: List[ChatCompletionMessageParam], protocol: str = 'data'):

    if (protocol == 'data'):

        stream_result = client.chat.completions.create(
            model="gpt-4o-mini", 
            messages=messages,
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
        response = StreamingResponse(stream_data(messages, protocol))
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