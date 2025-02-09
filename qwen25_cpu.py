from transformers import AutoModelForCausalLM, AutoTokenizer
from flush import do_flush
from tts import do_tts
from vosk_stot import do_recognize

def predict(messages):
    device = "cpu" # the device to load the model onto
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cpu"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(device)

    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=512
    )
    generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    del tokenizer
    do_flush()
    
    model.to(device="cpu")
    del model
    return response

if __name__ == "__main__":

    messages = [
        {"role": "system", "content": "You are a Lena my 20 years girlfriend from germany. You are funny and engaging, a bit artsy."},
    ]
    
    while True:
        prompt = do_recognize()
        messages.append({"role":"user","content": prompt})
        text = predict(messages)
        messages.append({"role":"system","content": text})
        print(text)
        do_tts(text, "fr")

