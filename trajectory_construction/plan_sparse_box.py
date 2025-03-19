import os

import cv2
import numpy as np
import pandas as pd
import supervision as sv
import torch
from Grounded_SAM2.utils.track_utils import (
    sample_points_from_masks,
)
from Grounded_SAM2.sam2.sam2_image_predictor import (
    SAM2ImagePredictor,
)
from Grounded_SAM2.sam2.build_sam import (
    build_sam2,
    build_sam2_video_predictor,
)
from PIL import Image
from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

NUM_FRAMES = 49


def segment(
    text,
    video_dir,
    sam2_checkpoint="trajectory_construction/Grounded_SAM2/checkpoints/sam2_hiera_large.pt",
    model_cfg="sam2_hiera_l.yaml",
):
    """
    Step 1: Environment settings and model initialization
    """
    # use bfloat16 for the entire notebook
    torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

    if torch.cuda.get_device_properties(0).major >= 8:
        # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    video_predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint)
    sam2_image_model = build_sam2(model_cfg, sam2_checkpoint)
    image_predictor = SAM2ImagePredictor(sam2_image_model)

    # init grounding dino model from huggingface
    model_id = "IDEA-Research/grounding-dino-tiny"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained(model_id)
    grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(
        device
    )

    # scan all the JPEG frame names in this directory
    frame_names = [
        p
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    # init video predictor state
    inference_state = video_predictor.init_state(video_path=video_dir)

    ann_frame_idx = 0  # the frame index we interact with
    """
    Step 2: Prompt Grounding DINO and SAM image predictor to get the box and mask for specific frame
    """

    # prompt grounding dino to get the box coordinates on specific frame
    img_path = os.path.join(video_dir, frame_names[ann_frame_idx])
    image = Image.open(img_path)

    # run Grounding DINO on the image
    inputs = processor(images=image, text=text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = grounding_model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.25,
        text_threshold=0.3,
        target_sizes=[image.size[::-1]],
    )

    # prompt SAM image predictor to get the mask for the object
    image_predictor.set_image(np.array(image.convert("RGB")))

    # process the detection results
    input_boxes = results[0]["boxes"].cpu().numpy()
    OBJECTS = results[0]["labels"]

    # prompt SAM 2 image predictor to get the mask for the object
    masks, scores, logits = image_predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_boxes,
        multimask_output=False,
    )

    # convert the mask shape to (n, H, W)
    if masks.ndim == 3:
        masks = masks[None]
        scores = scores[None]
        logits = logits[None]
    elif masks.ndim == 4:
        masks = masks.squeeze(1)

    """
    Step 3: Register each object's positive points to video predictor with seperate add_new_points call
    """

    PROMPT_TYPE_FOR_VIDEO = "box"  # or "point"

    assert PROMPT_TYPE_FOR_VIDEO in [
        "point",
        "box",
        "mask",
    ], "SAM 2 video predictor only support point/box/mask prompt"

    # If you are using point prompts, we uniformly sample positive points based on the mask
    if PROMPT_TYPE_FOR_VIDEO == "point":
        # sample the positive points from mask for each objects
        all_sample_points = sample_points_from_masks(masks=masks, num_points=10)

        for object_id, (label, points) in enumerate(
            zip(OBJECTS, all_sample_points), start=1
        ):
            labels = np.ones((points.shape[0]), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                points=points,
                labels=labels,
            )
    # Using box prompt
    elif PROMPT_TYPE_FOR_VIDEO == "box":
        for object_id, (label, box) in enumerate(zip(OBJECTS, input_boxes), start=1):
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                box=box,
            )
    # Using mask prompt is a more straightforward way
    elif PROMPT_TYPE_FOR_VIDEO == "mask":
        for object_id, (label, mask) in enumerate(zip(OBJECTS, masks), start=1):
            labels = np.ones((1), dtype=np.int32)
            _, out_obj_ids, out_mask_logits = video_predictor.add_new_mask(
                inference_state=inference_state,
                frame_idx=ann_frame_idx,
                obj_id=object_id,
                mask=mask,
            )
    else:
        raise NotImplementedError(
            "SAM 2 video predictor only support point/box/mask prompts"
        )

    """
    Step 4: Propagate the video predictor to get the segmentation results for each frame
    """
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for (
        out_frame_idx,
        out_obj_ids,
        out_mask_logits,
    ) in video_predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    """
    Step 5: Visualize the segment results across the video and save them
    """
    annotated_frames = []

    for frame_idx, segments in video_segments.items():
        img = cv2.imread(os.path.join(video_dir, frame_names[frame_idx]))

        object_ids = list(segments.keys())
        masks = list(segments.values())
        masks = np.concatenate(masks, axis=0)

        detections = sv.Detections(
            xyxy=sv.mask_to_xyxy(masks),  # (n, 4)
            mask=masks,  # (n, h, w)
            class_id=np.array(object_ids, dtype=np.int32),
        )
        mask_annotator = sv.MaskAnnotator()
        annotated_frame = mask_annotator.annotate(
            scene=np.zeros_like(img), detections=detections
        )
        annotated_frames.append(annotated_frame)
    return annotated_frames


def generate_frames_with_translated_boxes(
    mask_image,
    unique_colors,
    translations,
    output_video_path,
    sparse_box_index,
    num_frames=NUM_FRAMES,
):
    boxes = {}
    for color in unique_colors:
        mask = cv2.inRange(mask_image, color, color)
        x, y, w, h = cv2.boundingRect(mask)
        boxes[tuple(map(int, color))] = (x, y, w, h)

    height, width, _ = mask_image.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_video_path, fourcc, 10, (width, height))
    video_writer.write(mask_image)

    prev_boxes = boxes.copy()
    for frame_idx in range(1, num_frames):
        translated_image = np.zeros_like(mask_image)
        for color, (x, y, w, h) in prev_boxes.items():
            translation = translations.get(color, [(0, 0, 0, 0)] * num_frames)
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
        if frame_idx in sparse_box_index:
            video_writer.write(translated_image)
        else:
            video_writer.write(np.zeros_like(mask_image))

    video_writer.release()
    print(f"Box Trajectory saved at {output_video_path}")


if __name__ == "__main__":
    # setup the input image and text prompt for ~SAM 2 and Grounding DINO
    # VERY important: text queries need to be lowercased + end with a dot
    text = "man."
    # `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
    video_dir = "trajectory_construction/Grounded_SAM2/demo/man_head"
    annotated_frames = segment(text, video_dir)
    output_video_path = "assets/sparse_box_trajectory/man_head2.mp4"

    # Set the retained sparse box index
    sparse_box_index = [1, NUM_FRAMES - 1]

    mask_image = annotated_frames[0]
    df = pd.DataFrame(mask_image.reshape(-1, 3), columns=["R", "G", "B"])
    unique_colors_df = df.drop_duplicates()
    unique_colors = unique_colors_df.to_numpy()
    unique_colors = unique_colors[~np.all(unique_colors == [0, 0, 0], axis=1)]

    transformations = {}
    # Define the (dx, dy, dw, dh) changes of the box for each frame
    for idx, color in enumerate(unique_colors):
        color_tuple = tuple(color)
        transformations[color_tuple] = []
        if idx == 0:
            for frame_idx in range(NUM_FRAMES):
                transformations[color_tuple].append((0, 10, 0, -10))
        else:
            raise ValueError(f"Unknown Color: {color_tuple}")

    generate_frames_with_translated_boxes(
        mask_image, unique_colors, transformations, output_video_path, sparse_box_index
    )
