
from openai import OpenAI

client = OpenAI()

def predict(messages):
    chart_completion = client.chat.completions.create(model="gpt-5.6", messages=messages)
    return chart_completion.choices[0].message.content
    

def generate_image(prompt):
    response = client.responses.create(
        model="gpt-5.6",
        input="Generate an image of gray tabby cat hugging an otter with an orange scarf",
        tools=[{"type": "image_generation", "action": "generate"}],
    )

    # Save the image to a file
    image_data = [
        output.result
        for output in response.output
        if output.type == "image_generation_call"
    ]

    return image_data[0] if image_data else None    

if __name__ == "__main__":

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
