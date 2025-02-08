import diffusers
import transformers
import bitsandbytes as bnb
from diffusers import FluxPipeline, FluxTransformer2DModel
from transformers import T5EncoderModel
import torch
import gc

def flush():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_peak_memory_stats()


def bytes_to_giga_bytes(bytes):
    return bytes / 1024 / 1024 / 1024

flush()

# Checkpoints
ckpt_id = "black-forest-labs/FLUX.1-dev"
ckpt_4bit_id = "hf-internal-testing/flux.1-dev-nf4-pkg"

prompt = '''
A cat.
'''

text_encoder_2_4bit = T5EncoderModel.from_pretrained(
    ckpt_4bit_id,
    subfolder="text_encoder_2",
).to(torch.device("cuda"))

pipeline = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev",
    text_encoder_2=text_encoder_2_4bit,
    transformer=None,
    vae=None,
    torch_dtype=torch.float16,
).to(torch.device("cuda"))

with torch.no_grad():
    prompt_embeds, pooled_prompt_embeds, text_ids = pipeline.encode_prompt(
        prompt=prompt, prompt_2=None, max_sequence_length=256
    )

pipeline.to(torch.device("cpu"))
del pipeline
flush()

transformer_4bit = FluxTransformer2DModel.from_pretrained(ckpt_4bit_id, subfolder="transformer",low_cpu_mem_usage=True).to(torch.device("cuda"))
pipeline = FluxPipeline.from_pretrained(
    ckpt_id,
    text_encoder=None,
    text_encoder_2=None,
    tokenizer=None,
    tokenizer_2=None,
    transformer=transformer_4bit,
    torch_dtype=torch.float16,
).to(torch.device("cuda"))

pipeline.enable_model_cpu_offload()

print("Running denoising.")
height, width = 1024, 1024
images = pipeline(
    prompt_embeds=prompt_embeds,
    pooled_prompt_embeds=pooled_prompt_embeds,
    num_inference_steps=50,
    guidance_scale=5.5,
    height=height,
    width=width,
    output_type="pil",
).images

# Display the image
images[0].save("flux-nf4-dev.png")
