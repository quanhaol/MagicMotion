# export CUDA_VISIBLE_DEVICES=0
MODEL_PATH="THUDM/CogVideoX-5b-I2V"
perception_head_path="ckpts/stage2/perception_head.pt"
controlnet_path="ckpts/stage2/trajectory_controlnet.pt"

python magicmotion/inference.py \
    --pretrained_model_name_or_path  $MODEL_PATH \
    --pretrained_controlnet_path $controlnet_path \
    --pretrained_perception_head_path $perception_head_path \
    --validation_args_csv  "magicmotion/validation_args/demo/demo_box.csv" \
    --num_validation_videos 1 \
    --height 480 \
    --width 720 \
    --use_perception_head
