from typing import Union

from fastapi import FastAPI
from pydantic import BaseModel
from qwen2 import predict
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import json

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class Messages(BaseModel):
    messages: List[Message]

@app.post("/chat")
def do_predict(input: Messages):
    print(json.dumps(input.model_dump(mode='python')['messages']))
    response = predict(input.model_dump(mode='python')['messages'])
    print(response)
    return {"reply": response}

    
