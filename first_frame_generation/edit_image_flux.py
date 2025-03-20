import torch
from diffusers import FluxControlPipeline, FluxTransformer2DModel
from diffusers.utils import load_image
from image_gen_aux import DepthPreprocessor
import argparse


def edit_image(prompt: str, ori_image_path: str, save_path: str):
    pipe = FluxControlPipeline.from_pretrained(
        "black-forest-labs/FLUX.1-Depth-dev", torch_dtype=torch.bfloat16
    ).to("cuda")

    control_image = load_image(ori_image_path)

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
    image.save(save_path)
    print(f"Edited image saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Edit an image using FLUX.")
    parser.add_argument(
        "--prompt", type=str, required=True, help="The prompt for image editing."
    )
    parser.add_argument(
        "--ori_image_path",
        type=str,
        required=True,
        help="The path to the original image.",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="The path to save the edited image.",
    )
    args = parser.parse_args()

    edit_image(args.prompt, args.ori_image_path, args.save_path)
