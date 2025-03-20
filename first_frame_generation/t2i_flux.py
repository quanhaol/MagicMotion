import torch
from diffusers import FluxPipeline
import argparse


def image_generation(prompt: str, save_path: str):
    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()

    out = pipe(
        prompt=prompt,
        guidance_scale=3.5,
        height=480,
        width=720,
        num_inference_steps=50,
    ).images[0]
    out.save(save_path)

    print(f"Generated image saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate an image using FLUX.")
    parser.add_argument(
        "--prompt", type=str, required=True, help="The prompt for image generation."
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="The path to save the generated image.",
    )
    args = parser.parse_args()

    pipe = FluxPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16
    )
    pipe.enable_model_cpu_offload()

    image_generation(args.prompt, args.save_path)
