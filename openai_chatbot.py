from tts import do_tts
from vosk_stot import do_recognize
import os
import json
from openai import OpenAI
import time
import re
from pathlib import Path
client = OpenAI()

def predict(messages):
    chart_completion = client.chat.completions.create(model="gpt-4o", messages=messages)
    return chart_completion.choices[0].message.content
    


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

    #for filename, code in matches:
    #    print(filename)
    #    print(code.strip())

    return matches

if __name__ == "__main__":

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


        if "#code" in prompt:
            prompt = prompt.replace("#code", create_code_hint("./output")) 
        messages.append({"role":"user","content": prompt})
        text = predict(messages)
        print(text)
        code_snippets = extract_code(text)
        for filename, code in code_snippets:
            Path(filename.rsplit('/',1)[0]).mkdir(parents=True,exist_ok=True)
            print(code,  file=open(filename, 'w'))
            text = text.replace(code, "")

        messages.append({"role":"assistant","content": text})
        
        with open("messages_history.json", "w") as jsonfile:
            jsonfile.write(json.dumps(messages))
        do_tts(text, "fr")
