# Trajectory Construction

## Mask Trajectory

We use Grounded-SAM2 to segment out mask trajectory from an **input video**.
Please refer to `plan_mask.py` for more details.

```bash
python trajectory_construction/plan_mask.py --text "elephant.rhino." --video_dir "trajectory_construction/Grounded_SAM2/demo/mammoth_rhino" --output_video_path "assets/mask_trajectory/mammoth_rhino.mp4"
```

## Box Trajectory

We use Grounded-SAM2 to segment out the first frame's mask from an **input image**.
Then we construct box trajectory by changing the box's (x, y, w, h) frame by frame.
Please refer to `plan_box.py` for more details.

```bash
python trajectory_construction/plan_box.py --text "head." --video_dir "trajectory_construction/Grounded_SAM2/demo/tiger" --output_video_path "assets/box_trajectory/tiger.mp4"
```

## Sparse Box Trajectory

We first construct box trajectory using the methods illustrated above, then only box trajectories in the specified index frames are retained.
See `plan_sparse_box.py` for more details.

```bash
python trajectory_construction/plan_sparse_box.py --text "man." --video_dir "trajectory_construction/Grounded_SAM2/demo/man_head" --output_video_path "assets/sparse_box_trajectory/man_head.mp4" --sparse_box_index 1 48
```
