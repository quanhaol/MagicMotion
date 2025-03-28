import argparse
import gc
import os
import tempfile
import threading
import time
import pandas as pd
import cv2
import gradio as gr
import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw
import shutil
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(project_root)
from trajectory_construction.Grounded_SAM2.sam2.build_sam import build_sam2
from trajectory_construction.Grounded_SAM2.sam2.sam2_image_predictor import (
    SAM2ImagePredictor,
)
from diffusers.models import AutoencoderKLCogVideoX
from diffusers.schedulers import CogVideoXDPMScheduler
from models.trajectory_controlnet import TrajectoryControlnet
from models.magicmotion_transformer_3d import MagicMotionTransformer3DModel
from pipelines.pipeline_magicmotion import MagicMotionPipeline
from transformers import T5EncoderModel, T5Tokenizer
from args import get_args
from diffusers.utils import export_to_video, load_image, load_video

PROVIDED_EXAMPLES = {
    "cartoon_wizard": [
        "assets/box_trajectory/cartoon_wizard.mp4",
        "assets/images/condition/cartoon_wizard.jpg",
        "Two cartoon wizards walking towards each other",
        42,
    ],
    "floatie": [
        "assets/box_trajectory/floatie.mp4",
        "assets/images/condition/floatie.jpg",
        "A unicorn floatie floating in a pool",
        18,
    ],
    "child_horse": [
        "assets/box_trajectory/child_horse.mp4",
        "assets/images/condition/child_horse.jpg",
        "horse moving in the sky with a child on its back",
        8,
    ],
    "priestess": [
        "assets/box_trajectory/priestess.mp4",
        "assets/images/condition/priestess.jpg",
        "A priestess lifting a ball to her head",
        42,
    ],
}
article = r"""
---

📝 **Citation**
<br>
```bibtex
@misc{li2025magicmotioncontrollablevideogeneration,
      title={MagicMotion: Controllable Video Generation with Dense-to-Sparse Trajectory Guidance},
      author={Quanhao Li and Zhen Xing and Rui Wang and Hui Zhang and Qi Dai and Zuxuan Wu},
      year={2025},
      eprint={2503.16421},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.16421},
}
```
"""
# init sam image predictor
sam2_checkpoint = (
    "trajectory_construction/Grounded_SAM2/checkpoints/sam2_hiera_large.pt"
)
model_cfg = "sam2_hiera_l.yaml"
sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
predictor = SAM2ImagePredictor(sam2_image_model)
SAM_labels = []
color_list = []
masks_selected = []
masks_selected_tmp = None
first_frame_path = "tmp/first_frame.png"
mask_image_path = "tmp/mask_image.png"
traj_video_path = "tmp/traj_video.mp4"
output_path = "tmp/output.mp4"
for i in range(20):
    color = np.concatenate([np.random.random(4) * 255], axis=0).astype(int)
    if color[-1] < 150:
        color[-1] = 150
    color_list.append(color)
mask_colors = []


def preprocess_first_frame(image, width=720, height=480):
    if not os.path.exists(first_frame_path):
        os.makedirs(os.path.dirname(first_frame_path), exist_ok=True)
    image_pil = Image.open(image.name)
    image_pil = image_pil.resize((width, height), Image.BILINEAR)
    image_pil.save(first_frame_path)
    print(f"Saved first frame to {first_frame_path}")
    global SAM_labels
    SAM_labels = []
    global mask_colors
    mask_colors = []
    global masks_selected
    masks_selected = []
    global masks_selected_tmp
    masks_selected_tmp = None
    return first_frame_path, gr.State([])


def image_click(click_points, segment_flag, foreground_flag, evt: gr.SelectData):
    click_points.value[-1].append(evt.index)
    # Segment Main Objects
    if segment_flag == 1:
        print("Segment Main Objects...")
        transparent_background = Image.open(first_frame_path).convert("RGBA")
        w, h = transparent_background.size
        image = cv2.imread(first_frame_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        predictor.set_image(image)

        input_point = []
        for point in click_points.value[-1]:
            input_point.append([point[0], point[1]])
        input_point = np.array(input_point)
        print(input_point)

        global SAM_labels
        if foreground_flag == "foreground_point":
            foreground_flag = 1
        else:
            foreground_flag = 0
        SAM_labels.append(foreground_flag)
        input_label = np.array(SAM_labels)

        masks, scores, logits = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        global masks_selected_tmp
        masks_selected_tmp = masks
        transparent_layer = 0
        for idx, mask in enumerate(masks_selected):
            color = color_list[idx]
            transparent_layer = (
                mask[1].reshape(h, w, 1) * color.reshape(1, 1, -1) + transparent_layer
            )

        color = color_list[len(masks_selected)]
        transparent_layer = (
            masks_selected_tmp[1].reshape(h, w, 1) * color.reshape(1, 1, -1)
            + transparent_layer
        )

        transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
        black_background = Image.new("RGB", transparent_layer.size, (0, 0, 0))

        rgb_layer = transparent_layer.convert("RGB")
        black_background.paste(rgb_layer, mask=transparent_layer.split()[-1])
        black_background.save(mask_image_path)

        mask_image = cv2.imread(mask_image_path)
        df = pd.DataFrame(mask_image.reshape(-1, 3), columns=["R", "G", "B"])
        unique_colors_df = df.drop_duplicates()
        unique_colors = unique_colors_df.to_numpy()
        unique_colors = unique_colors[~np.all(unique_colors == [0, 0, 0], axis=1)]
        for color in unique_colors:
            color_tuple = tuple(color)
            if color_tuple not in mask_colors:
                mask_colors.append(color_tuple)
        print("Unique colors in the mask image:", mask_colors)

        alpha_coef = 0.99
        im2_data = transparent_layer.getdata()
        new_im2_data = [(r, g, b, int(a * alpha_coef)) for r, g, b, a in im2_data]
        transparent_layer.putdata(new_im2_data)
        trajectory_map = Image.alpha_composite(
            transparent_background, transparent_layer
        )

    # Draw Trajectory
    else:
        print("Draw Trajectory...")
        transparent_background = Image.open(first_frame_path).convert("RGBA")
        w, h = transparent_background.size
        transparent_layer = 0
        for idx, mask in enumerate(masks_selected):
            color = color_list[idx]
            transparent_layer = (
                mask[1].reshape(h, w, 1) * color.reshape(1, 1, -1) + transparent_layer
            )

        for idx, track in enumerate(click_points.value):
            mask = np.zeros((h, w, 3))
            color = color_list[idx + 1]
            transparent_layer = (
                mask[:, :, 0].reshape(h, w, 1) * color.reshape(1, 1, -1)
                + transparent_layer
            )

            if len(track) > 1:
                for i in range(len(track) - 1):
                    start_point = track[i]
                    end_point = track[i + 1]
                    vx = end_point[0] - start_point[0]
                    vy = end_point[1] - start_point[1]
                    arrow_length = np.sqrt(vx**2 + vy**2)
                    if i == len(track) - 2:
                        cv2.arrowedLine(
                            transparent_layer,
                            tuple(start_point),
                            tuple(end_point),
                            (255, 0, 0, 255),
                            2,
                            tipLength=8 / arrow_length,
                        )
                    else:
                        cv2.line(
                            transparent_layer,
                            tuple(start_point),
                            tuple(end_point),
                            (255, 0, 0, 255),
                            2,
                        )
            else:
                cv2.circle(transparent_layer, tuple(track[0]), 5, (255, 0, 0, 255), -1)

        transparent_layer = Image.fromarray(transparent_layer.astype(np.uint8))
        alpha_coef = 0.99
        im2_data = transparent_layer.getdata()
        new_im2_data = [(r, g, b, int(a * alpha_coef)) for r, g, b, a in im2_data]
        transparent_layer.putdata(new_im2_data)

        trajectory_map = Image.alpha_composite(
            transparent_background, transparent_layer
        )
    return click_points, trajectory_map


def start_segment(click_points):
    global SAM_labels
    SAM_labels = []
    click_points.value.append([])
    return click_points, 1


def start_traj(click_points):
    click_points.value[-1] = []
    masks_selected.append(masks_selected_tmp)
    return click_points, 0


def process_points(points, frames=49):
    defualt_points = [[128, 128]] * frames

    if len(points) < 2:
        return defualt_points

    elif len(points) >= frames:
        skip = len(points) // frames
        return points[::skip][: frames - 1] + points[-1:]
    else:
        insert_num = frames - len(points)
        insert_num_dict = {}
        interval = len(points) - 1
        n = insert_num // interval
        m = insert_num % interval
        for i in range(interval):
            insert_num_dict[i] = n
        for i in range(m):
            insert_num_dict[i] += 1

        res = []
        for i in range(interval):
            insert_points = []
            x0, y0 = points[i]
            x1, y1 = points[i + 1]

            delta_x = x1 - x0
            delta_y = y1 - y0
            for j in range(insert_num_dict[i]):
                x = x0 + (j + 1) / (insert_num_dict[i] + 1) * delta_x
                y = y0 + (j + 1) / (insert_num_dict[i] + 1) * delta_y
                insert_points.append([int(x), int(y)])

            res += points[i : i + 1] + insert_points
        res += points[-1:]
        return res


def process_points(points, frames=49):
    defualt_points = [[128, 128]] * frames

    if len(points) < 2:
        return defualt_points

    elif len(points) >= frames:
        skip = len(points) // frames
        return points[::skip][: frames - 1] + points[-1:]
    else:
        insert_num = frames - len(points)
        insert_num_dict = {}
        interval = len(points) - 1
        n = insert_num // interval
        m = insert_num % interval
        for i in range(interval):
            insert_num_dict[i] = n
        for i in range(m):
            insert_num_dict[i] += 1

        res = []
        for i in range(interval):
            insert_points = []
            x0, y0 = points[i]
            x1, y1 = points[i + 1]

            delta_x = x1 - x0
            delta_y = y1 - y0
            for j in range(insert_num_dict[i]):
                x = x0 + (j + 1) / (insert_num_dict[i] + 1) * delta_x
                y = y0 + (j + 1) / (insert_num_dict[i] + 1) * delta_y
                insert_points.append([int(x), int(y)])

            res += points[i : i + 1] + insert_points
        res += points[-1:]
        return res


def fn_vis_traj(traj_list, num_frames=49):
    mask_image = cv2.imread(mask_image_path)

    imgs = []
    imgs.append(mask_image)
    boxes = {}
    transformations = {}

    for idx, traj in enumerate(traj_list.value):
        color = np.array(mask_colors[idx], dtype=np.uint8)
        mask = cv2.inRange(mask_image, color, color)
        x, y, w, h = cv2.boundingRect(mask)
        print(f"Initial box for color {color}: x={x}, y={y}, w={w}, h={h}")
        boxes[tuple(map(int, color))] = (x, y, w, h)
        processed_points = process_points(traj, frames=num_frames)
        print(f"Processed points for color {color}: {processed_points}")

        frame_transforms = []
        for i in range(len(processed_points) - 1):
            start_point = processed_points[i]
            end_point = processed_points[i + 1]
            dx = end_point[0] - start_point[0]
            dy = end_point[1] - start_point[1]
            frame_transforms.append((dx, dy, 0, 0))  # dw 和 dh 默认为 0
        transformations[tuple(map(int, color))] = frame_transforms

    prev_boxes = boxes.copy()
    for frame_idx in range(0, num_frames - 1):
        translated_image = np.zeros_like(mask_image)
        for color, (x, y, w, h) in prev_boxes.items():
            translation = transformations.get(color, [(0, 0, 0, 0)] * num_frames)
            dx, dy, dw, dh = translation[frame_idx]
            new_x, new_y = x + dx, y + dy
            new_w, new_h = w + dw, h + dh
            cv2.rectangle(
                translated_image,
                (new_x, new_y),
                (new_x + new_w, new_y + new_h),
                color,
                thickness=min(mask_image.shape[0], mask_image.shape[1]) // 100,
            )
            prev_boxes[color] = (new_x, new_y, new_w, new_h)

        imgs.append(translated_image)

    fps = 8
    writer = imageio.get_writer(traj_video_path, format="mp4", mode="I", fps=fps)

    for img in imgs:
        if len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        writer.append_data(img)

    writer.close()

    return traj_video_path


def model_run(prompt, seed, fps=8, num_frames=49, weight=1.0):
    global model_args
    global pipe
    global model_config

    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pipeline_args = {
        "image": load_image(first_frame_path),
        "prompt": prompt,
        "guidance_scale": model_args.guidance_scale,
        "use_dynamic_cfg": model_args.use_dynamic_cfg,
        "height": 480,
        "width": 720,
        "max_sequence_length": model_config.max_text_seq_length,
        "trajectory_maps": load_video(traj_video_path),
        "controlnet_weights": float(weight),
    }
    print("Generating:", prompt)
    video_generate = pipe(
        **pipeline_args,
        num_frames=num_frames,
        generator=torch.Generator().manual_seed(seed),
        output_type="np",
    ).frames[0]

    export_to_video(video_generate, output_path, fps=fps)

    return output_path


def load_pipe(args):
    global pipe
    global model_config
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

    # 3. Enable CPU offload for the model.
    # turn off if you have multiple GPUs or enough GPU memory(such as H100) and it will cost less time in inference
    # and enable to("cuda")

    pipe.to("cuda")
    # pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()


def add_provided_example(example_name):
    provided_traj_path = PROVIDED_EXAMPLES[example_name][0]
    provided_first_frame_path = PROVIDED_EXAMPLES[example_name][1]
    provided_prompt = PROVIDED_EXAMPLES[example_name][2]
    provided_seed = PROVIDED_EXAMPLES[example_name][3]
    os.makedirs(os.path.dirname(traj_video_path), exist_ok=True)
    os.makedirs(os.path.dirname(first_frame_path), exist_ok=True)
    shutil.copy(provided_traj_path, traj_video_path)
    shutil.copy(provided_first_frame_path, first_frame_path)
    return traj_video_path, first_frame_path, provided_prompt, provided_seed


def main(args):
    global canvas_width, canvas_height
    canvas_width, canvas_height = 720, 480

    demo = gr.Blocks()
    with demo:
        gr.Markdown(
            """
            <div style="text-align: center; font-size: 32px; font-weight: bold; margin-bottom: 20px;">
                MagicMotion: Controllable Video Generation with Dense-to-Sparse Trajectory Guidance
            </div>
            <div style="text-align: center;font-size: 20px;">
                <a href="https://github.com/quanhaol/MagicMotion">Github</a> |
                <a href="https://quanhaol.github.io/magicmotion-site/">Project Page</a> |
                <a href="https://arxiv.org/abs/2503.16421">arXiv</a>
            </div>
            """
        )
        segment_flag = gr.State()
        click_points = gr.State([])

        with gr.Column():
            with gr.Row():
                with gr.Column():
                    # step1 - Segment out Moving Object
                    gr.Markdown(
                        "---\n## Step 1/3: Segment the Moving Object",
                        show_label=False,
                        visible=True,
                    )
                    gr.Markdown(
                        "(Currently, this demo supports only MagicMotion Stage 2 and is limited to box translation without box shape changes. \
                        To explore the full version of MagicMotion, please use the command line provided on <a href='https://github.com/quanhaol/MagicMotion'>GitHub</a>.) \
                        \n 1. Upload the start image via the `Upload Start Image` button. \
                        \n 2. **Click the `Select Area with SAM` button.** \
                        \n 3. **Click on `Point Labels` to determine whether the current input point is in foreground or background.** \
                        \n 4. Click on `Start Frame` to segment out main objects.",
                        show_label=False,
                        visible=True,
                    )
                    foreground_flag = gr.Radio(
                        choices=["foreground_point", "background_point"],
                        value="foreground_point",
                        label="Point Labels",
                        interactive=True,
                        visible=True,
                    )
                    with gr.Row():
                        image_upload_button = gr.UploadButton(
                            label="Upload Start Image", file_types=["image"]
                        )
                        select_area_button = gr.Button(value="Select Area with SAM")
                    # step2 - object motion control - draw yourself
                    gr.Markdown(
                        "---\n## Step 2/3: Draw A Trajectory",
                        show_label=False,
                        visible=True,
                    )
                    gr.Markdown(
                        "\n 1. **Click the `Add New Trajectory` button.** \
                        \n 2. Click on the `Start Frame` to create a trajectory. Each click adds a new point to the trajectory. \
                        \n 3. Click on `Visualize Trajectory` to generate trajectory video. \
                        \n 4. Return to Step1 if you want to add another moving object and trajectory.",
                        show_label=False,
                        visible=True,
                    )

                    with gr.Row():
                        add_drag_button = gr.Button(value="Add New Trajectory")
                        traj_vis = gr.Button(value="Visualize Trajectory", visible=True)

                with gr.Column():
                    input_image = gr.Image(
                        label="Start Frame",
                        interactive=True,
                        height=canvas_height,
                        width=canvas_width,
                    )

                    vis_traj = gr.Video(
                        value=None,
                        label="Trajectory",
                        visible=True,
                        width=canvas_width,
                        height=canvas_height,
                    )

            # step2 - Add prompt and seed to generate videos
            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        "---\n## Step 3/3: Add prompt and seed",
                        show_label=False,
                        visible=True,
                    )
                    prompt = gr.Textbox(
                        value="", label="Prompt", interactive=True, visible=True
                    )
                    seed = gr.Number(
                        value=42,
                        precision=0,
                        interactive=True,
                        label="Seed",
                        visible=True,
                    )
                    start = gr.Button(value="Generate", visible=True)
                with gr.Column():
                    gen_video = gr.Video(
                        value=None, label="Generate Video", visible=True
                    )

            # traj examples
            with gr.Column():
                gr.Markdown("---\n## Examples", show_label=False, visible=True)
                with gr.Row():
                    example_1 = gr.Button(value="cartoon_wizard", visible=True)
                    example_2 = gr.Button(value="floatie", visible=True)
                    example_3 = gr.Button(value="child_horse", visible=True)
                    example_4 = gr.Button(value="priestess", visible=True)

        example_1.click(
            fn=add_provided_example,
            inputs=[example_1],
            outputs=[vis_traj, input_image, prompt, seed],
        )
        example_2.click(
            fn=add_provided_example,
            inputs=[example_2],
            outputs=[vis_traj, input_image, prompt, seed],
        )
        example_3.click(
            fn=add_provided_example,
            inputs=[example_3],
            outputs=[vis_traj, input_image, prompt, seed],
        )
        example_4.click(
            fn=add_provided_example,
            inputs=[example_4],
            outputs=[vis_traj, input_image, prompt, seed],
        )

        image_upload_button.upload(
            preprocess_first_frame,
            image_upload_button,
            [input_image, click_points],
        )
        select_area_button.click(
            start_segment, click_points, [click_points, segment_flag]
        )

        add_drag_button.click(start_traj, click_points, [click_points, segment_flag])

        input_image.select(
            fn=image_click,
            inputs=[click_points, segment_flag, foreground_flag],
            outputs=[click_points, input_image],
        )
        traj_vis.click(
            fn=fn_vis_traj,
            inputs=click_points,
            outputs=[vis_traj],
        )
        start.click(
            fn=model_run,
            inputs=[prompt, seed],
            outputs=gen_video,
        )
        gr.Markdown(article)

    demo.queue(max_size=32).launch(**args)


if __name__ == "__main__":
    global model_args
    model_args = get_args()
    load_pipe(model_args)
    print("******************** model loaded ********************")

    launch_kwargs = {
        "server_name": "0.0.0.0" if "SPACE_ID" in os.environ else "127.0.0.1",
        "server_port": 7860,
        "inbrowser": False,
        "share": False,
    }

    main(launch_kwargs)
