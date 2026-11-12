#!/bin/bash
# DreamZero DROID action training with DROID video-only cotrain side branch.

export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

# ============ USER CONFIGURATION ============
DROID_DATA_ROOT=${DROID_DATA_ROOT:?Set DROID_DATA_ROOT}
OUTPUT_DIR=${OUTPUT_DIR:?Set OUTPUT_DIR}

PER_DEVICE_BS=${PER_DEVICE_BS:-1}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
NUM_GPUS=${NUM_GPUS:-8}
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE:-$((NUM_GPUS * PER_DEVICE_BS))}
COTRAIN_CYCLE_LENGTH=${COTRAIN_CYCLE_LENGTH:-1}
COTRAIN_VIDEO_BATCHES_PER_CYCLE=${COTRAIN_VIDEO_BATCHES_PER_CYCLE:-1}
COTRAIN_VIDEO_LOSS_WEIGHT=${COTRAIN_VIDEO_LOSS_WEIGHT:-1.0}
COTRAIN_ACTION_LOSS_WEIGHT=${COTRAIN_ACTION_LOSS_WEIGHT:-1.0}

WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"./checkpoints/Wan2.2-TI2V-5B"}
IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
# =============================================

# ============ AUTO-DOWNLOAD WEIGHTS ============
if [ ! -d "$WAN22_CKPT_DIR" ] || [ -z "$(ls -A "$WAN22_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.2-TI2V-5B not found at $WAN22_CKPT_DIR. Downloading from HuggingFace..."
    huggingface-cli download Wan-AI/Wan2.2-TI2V-5B --local-dir "$WAN22_CKPT_DIR"
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi

if [ ! -f "$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth" ]; then
    echo "Image encoder not found. Downloading Wan2.1-I2V-14B-480P (for CLIP only)..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$IMAGE_ENCODER_DIR"
fi
# ================================================

# Validate dataset exists
if [ ! -d "$DROID_DATA_ROOT" ]; then
    echo "ERROR: DROID dataset not found at $DROID_DATA_ROOT"
    exit 1
fi

# Cotrain contract:
# - labeled branch: DROID, action_horizon=24, action/state loss enabled
# - video side branch: DROID video-only, action_horizon=0, action loss must be zero
# - both branches use the DROID 3-view layout configured by their data configs
# ============ LAUNCH TRAINING ============
torchrun --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    trainer._target_=groot.vla.experiment.base.CoTrainTrainer \
    report_to=tensorboard \
    data=dreamzero/droid_relative_wan22 \
    video_dataset=dreamzero/droid_cotrain \
    cotrain.enabled=true \
    cotrain.video_cycle_length="$COTRAIN_CYCLE_LENGTH" \
    cotrain.video_batches_per_cycle="$COTRAIN_VIDEO_BATCHES_PER_CYCLE" \
    cotrain.video_loss_weight="$COTRAIN_VIDEO_LOSS_WEIGHT" \
    cotrain.action_loss_weight="$COTRAIN_ACTION_LOSS_WEIGHT" \
    wandb_project=dreamzero \
    train_architecture=full \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan22 \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=$LEARNING_RATE \
    +training_args.max_grad_norm=1.0 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=$PER_DEVICE_BS \
    global_batch_size=$GLOBAL_BATCH_SIZE \
    save_steps=10000 \
    max_steps=200000 \
    weight_decay=1e-5 \
    save_total_limit=30 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=true \
    dataloader_num_workers=4 \
    image_resolution_width=320 \
    image_resolution_height=160 \
    save_lora_only=false \
    max_chunk_size=4 \
    save_strategy=steps \
    droid_data_root=$DROID_DATA_ROOT \
    dit_version=$WAN22_CKPT_DIR \
    text_encoder_pretrained_path=$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN22_CKPT_DIR/Wan2.2_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR
