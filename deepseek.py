import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import gc

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()


def predict(text):
    model_name = "deepseek-ai/deepseek-llm-7b-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    model.generation_config = GenerationConfig.from_pretrained(model_name)
    model.generation_config.pad_token_id = model.generation_config.eos_token_id

    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs.to(model.device), max_new_tokens=100)

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    #model.to(torch.device("cpu"))
    del inputs
    del model
    flush()
    return result


if __name__ == "__main__":
    text = '''
    Hello, who are you ?
    '''

    reply = predict(text)
    print(reply)
