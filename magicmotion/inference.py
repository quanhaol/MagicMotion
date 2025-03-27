# Copyright 2024 The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pandas as pd
import torch
from args import get_args
from diffusers.models import AutoencoderKLCogVideoX
from diffusers.schedulers import CogVideoXDPMScheduler
from diffusers.utils import export_to_video, load_image, load_video
from models.trajectory_controlnet import TrajectoryControlnet
from models.magicmotion_transformer_3d import MagicMotionTransformer3DModel
from pipelines.pipeline_magicmotion import MagicMotionPipeline
from transformers import T5EncoderModel, T5Tokenizer


def main(args):

    tokenizer = T5Tokenizer.from_pretrained(
        "THUDM/CogVideoX-5b-I2V", subfolder="tokenizer"
    )
    text_encoder = T5EncoderModel.from_pretrained(
        "THUDM/CogVideoX-5b-I2V", subfolder="text_encoder"
    ).cuda()
    vae = AutoencoderKLCogVideoX.from_pretrained(
        "THUDM/CogVideoX-5b-I2V", subfolder="vae"
    ).cuda()
    load_dtype = (
        torch.bfloat16
        if "5b" in args.pretrained_model_name_or_path.lower()
        else torch.float16
    )
    transformer = MagicMotionTransformer3DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        torch_dtype=load_dtype,
        revision=args.revision,
        variant=args.variant,
        use_perception_head=args.use_perception_head,
    )
    if os.path.exists(args.pretrained_perception_head_path):
        ckpt = torch.load(
            args.pretrained_perception_head_path, map_location="cpu", weights_only=False
        )
        perception_head_state_dict = {}
        for name, params in ckpt["state_dict"].items():
            perception_head_state_dict[name] = params
        m, u = transformer.perception_head.load_state_dict(
            perception_head_state_dict, strict=False
        )
        print(
            f"[ Weights from pretrained perception_head was loaded into transformer ] [M: {len(m)} | U: {len(u)}]"
        )
    model_config = (
        transformer.module.config
        if hasattr(transformer, "module")
        else transformer.config
    )
    controlnet_config = {}
    for k, v in model_config.items():
        if "use_perception_head" not in k:
            controlnet_config[k] = v
    controlnet = TrajectoryControlnet(
        **controlnet_config,
    )

    assert args.pretrained_controlnet_path is not None
    ckpt = torch.load(
        args.pretrained_controlnet_path, map_location="cpu", weights_only=False
    )
    controlnet_state_dict = {}
    for name, params in ckpt["state_dict"].items():
        controlnet_state_dict[name] = params
    m, u = controlnet.load_state_dict(controlnet_state_dict, strict=False)
    print(
        f"[ Weights from pretrained controlnet was loaded into controlnet ] [M: {len(m)} | U: {len(u)}]"
    )

    scheduler = CogVideoXDPMScheduler.from_pretrained(
        "THUDM/CogVideoX-5b-I2V", subfolder="scheduler"
    )
    pipe = MagicMotionPipeline(
        vae=vae,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        transformer=transformer,
        controlnet=controlnet,
        scheduler=scheduler,
    ).to(torch.bfloat16)

    num_frames = 49
    fps = 8

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")

    pipe.to("cuda")
    # pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    # Run inference
    if args.validation_args_csv and args.num_validation_videos > 0:
        validation_args = pd.read_csv(args.validation_args_csv)
        validation_prompts = validation_args["validation_prompt"].tolist()
        validation_images = validation_args["validation_images"].tolist()
        validation_trajectory_maps = validation_args[
            "validation_trajectory_maps"
        ].tolist()
        output_paths = validation_args["output_path"].tolist()
        controlnet_weights = validation_args["controlnet_weights"].tolist()
        seeds = validation_args["seed"].tolist()
        for (
            validation_image,
            validation_prompt,
            validation_trajectory_map,
            output_path,
            weight,
            seed,
        ) in zip(
            validation_images,
            validation_prompts,
            validation_trajectory_maps,
            output_paths,
            controlnet_weights,
            seeds,
        ):
            if os.path.exists(output_path):
                print(f"Output path {output_path} already exists. Skipping.")
                continue
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            pipeline_args = {
                "image": load_image(validation_image),
                "prompt": validation_prompt,
                "guidance_scale": args.guidance_scale,
                "use_dynamic_cfg": args.use_dynamic_cfg,
                "height": args.height,
                "width": args.width,
                "max_sequence_length": model_config.max_text_seq_length,
                "trajectory_maps": load_video(validation_trajectory_map),
                "controlnet_weights": float(weight),
            }
            print("Generating:", validation_prompt)
            video_generate = pipe(
                **pipeline_args,
                num_frames=num_frames,
                generator=torch.Generator().manual_seed(seed),
                output_type="np",
            ).frames[0]

            export_to_video(video_generate, output_path, fps=fps)


if __name__ == "__main__":
    args = get_args()
    main(args)
