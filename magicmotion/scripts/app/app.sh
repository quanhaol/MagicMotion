export CUDA_VISIBLE_DEVICES=0
MODEL_PATH="THUDM/CogVideoX-5b-I2V"
perception_head_path="ckpts/stage2/box_perception_head.pt"
controlnet_path="ckpts/stage2/box.pt"

python magicmotion/app.py \
    --pretrained_model_name_or_path  $MODEL_PATH \
    --pretrained_controlnet_path $controlnet_path \
    --pretrained_perception_head_path $perception_head_path \
    --use_perception_head
