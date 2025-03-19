# Trajectory Construction

## Mask Trajectory

We use Grounded-SAM2 to segment out mask trajectory from an **input video**.
Please refer to `plan_mask.py` for more details.

## Box Trajectory

We use Grounded-SAM2 to segment out the first frame's mask from an **input image**.
Then we construct box trajectory by changing the box's (x, y, w, h) frame by frame.
Please refer to `plan_box.py` for more details.

## Sparse Box Trajectory

We first construct box trajectory using the methods illustrated above, then only box trajectories in the specified index frames are retained.
See `plan_sparse_box.py` for more details.
