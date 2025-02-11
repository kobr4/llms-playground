from tts import do_tts
from vosk_stot import do_recognize
import os
import json
from openai import OpenAI
client = OpenAI()

def predict(messages):
    chart_completion = client.chat.completions.create(model="gpt-4o", messages=messages)
    return chart_completion.choices[0].message.content
    


if __name__ == "__main__":

    if not os.path.exists("messages_history.json"):
        messages = [
            {"role": "system", "content": "You are a Lena my 20 years girlfriend from germany. You are funny and engaging, a bit artsy."},
        ]
    else:
        with open("messages_history.json", "r") as jsonfile:
            messages = json.loads(jsonfile.read())
    
    while True:
        prompt = do_recognize()
        messages.append({"role":"user","content": prompt})
        text = predict(messages)
        messages.append({"role":"lena","content": text})
        print(text)
        with open("messages_history.json", "w") as jsonfile:
            jsonfile.write(json.dumps(messages))
        do_tts(text, "fr")