# export CUDA_VISIBLE_DEVICES=0
MODEL_PATH="THUDM/CogVideoX-5b-I2V"
controlnet_path="/datadrive2/cogvideox/mask/Pexels_MeViS_MOSE_DAVIS/Controlnet/checkpoint-17000.pt"

python magicmotion/inference.py \
    --pretrained_model_name_or_path  $MODEL_PATH \
    --pretrained_controlnet_path $controlnet_path \
    --validation_args_csv  "magicmotion/validation_args/demo/demo_mask.csv" \
    --num_validation_videos 1 \
    --height 480 \
    --width 720 \
