import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
)
pipe.enable_model_cpu_offload()

prompt = "A tiger sitting, head facing camera."
out = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    height=480,
    width=720,
    num_inference_steps=50,
).images[0]
out.save("assets/images/condition/tiger.png")
