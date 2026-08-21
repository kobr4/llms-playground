import os
import json    
import time
import re
from pathlib import Path
import pyscreenshot
from io import BytesIO
import base64    
from tts import do_tts
from vosk_stot import do_recognize
from openai_chatbot import predict, generate_image
from llm_chat_utils import generate_filename, classify

def create_capture():
    image = pyscreenshot.grab(backend="mss")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return base64_image
    

def create_code_hint(directory):
    code_hint = f"""
    - code is in directory {directory}
    - code should be between "[CODE(filename='filename')]" and "[/CODE]" (note the / to signify its closing) 
    - filename indicates the name of the file
    """
    return code_hint

def extract_code(input):
    matches = re.findall(
        r"\[CODE\(filename=['\"](.*?)['\"]\)\](.*?)\[/CODE\]",
        text,
        re.DOTALL
    )

    return matches

if not os.path.exists("messages_history.json"):
    messages = [
        {"role": "system", "content": "You are a Lena my tech savvy assistant. You are funny and engaging."},
    ]
else:
    with open("messages_history.json", "r") as jsonfile:
        messages = json.loads(jsonfile.read())

while True:
    prompt = do_recognize()
    if prompt == False:
        time.sleep(2)
        prompt = input("Keyboard input:")

    if "#code" not in prompt and "#capture" not in prompt and "#image" not in prompt:
        hashtag = classify(prompt)
        print(f"Detected hashtag: {hashtag}")
        if "#normal" not in hashtag:
            prompt = f"{prompt} {hashtag}" 

    if "#code" in prompt:
        prompt = prompt.replace("#code", create_code_hint("./output"))

    if "#capture" in prompt:
        prompt = prompt.replace("#capture", prompt)
        prompt = [ 
            {"type": "text", "text": prompt},
            {
                "type": "image_url",
                "image_url": {"url":f"data:image/jpeg;base64,{create_capture()}", "detail": "high"},
            }]

    if "#image" in prompt:
        image_base64 = generate_image(prompt)
        if image_base64:
            filename = generate_filename(prompt)
            with open(filename, "wb") as f:
                f.write(base64.b64decode(image_base64))
        
    
    messages.append({"role":"user","content": prompt})
    text = predict(messages)
    print(text)
    code_snippets = extract_code(text)
    for filename, code in code_snippets:
        if "/" in filename:
            Path(filename.rsplit('/',1)[0]).mkdir(parents=True,exist_ok=True)
        print(code,  file=open(filename, 'w'))
        text = text.replace(code, "")

    
    if not isinstance(messages[-1]['content'], str) :
        del messages[-1]
    
    messages.append({"role":"assistant","content": text})
    
    with open("messages_history.json", "w") as jsonfile:
        jsonfile.write(json.dumps(messages))
    do_tts(text, "fr")    