# !pip install git+https://github.com/huggingface/image_gen_aux
import torch
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from image_gen_aux import DepthPreprocessor

pipe = FluxControlPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Depth-dev", torch_dtype=torch.bfloat16
).to("cuda")

prompt = "A royal camel walking inside a palace."
control_image = load_image("assets/images/ori_image/camel.jpg")

processor = DepthPreprocessor.from_pretrained("LiheYoung/depth-anything-large-hf")
control_image = processor(control_image)[0].convert("RGB")

image = pipe(
    prompt=prompt,
    control_image=control_image,
    height=480,
    width=854,
    num_inference_steps=30,
    guidance_scale=10.0,
    generator=torch.Generator().manual_seed(42),
).images[0]
image.save(f"assets/images/condition/camel_royal.png")
