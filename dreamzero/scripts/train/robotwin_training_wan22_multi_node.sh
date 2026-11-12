#!/bin/bash
# DreamZero RoboTwin Multi-Node Training Script with Wan2.2-TI2V-5B backbone
#
# Usage:
#   bash scripts/train/robotwin_training_wan22_lr1e_4_multi_node.sh
#
# Prerequisites:
#   - RoboTwin dataset in LeRobot format with modality.json injected
#     (Run: python scripts/prepare_robotwin_modality.py --data_root $ROBOTWIN_DATA_ROOT)
#   - Wan2.2-TI2V-5B weights (download from HuggingFace)
#   - Image encoder (CLIP) from Wan2.1
#   - umt5-xxl tokenizer

export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1
export WANDB_MODE=offline

# ============ USER CONFIGURATION ============
ROBOTWIN_DATA_ROOT=${ROBOTWIN_DATA_ROOT:-"./data/robotwin-clean-and-aug-lerobot"}
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_robotwin_wan22_lr1e_4_multi_node"}

# multi-node
PET_NPROC_PER_NODE=${PET_NPROC_PER_NODE:?must set PET_NPROC_PER_NODE}
PET_NNODES=${PET_NNODES:?must set PET_NNODES}
PET_NODE_RANK=${PET_NODE_RANK:?must set PET_NODE_RANK}
MASTER_ADDR=${MASTER_ADDR:?must set MASTER_ADDR}
MASTER_PORT=${MASTER_PORT:?must set MASTER_PORT}

# Wan2.2-TI2V-5B checkpoint (contains: diffusion weights, T5, VAE)
WAN22_CKPT_DIR=${WAN22_CKPT_DIR:-"./checkpoints/Wan2.2-TI2V-5B"}
# Image encoder: Wan2.2-TI2V-5B does NOT include CLIP - use Wan2.1's or standalone
IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}

# Which subsets to include (space-separated)
SUBSETS=${SUBSETS:-"lerobot_robotwin_eef_aug_500 lerobot_robotwin_eef_clean_50"}
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
if [ ! -d "$ROBOTWIN_DATA_ROOT" ]; then
    echo "ERROR: RoboTwin dataset not found at $ROBOTWIN_DATA_ROOT"
    echo "Please set ROBOTWIN_DATA_ROOT to the directory containing lerobot_robotwin_eef_aug_500/ etc."
    exit 1
fi

# ============ AUTO-DISCOVER TASK PATHS ============
echo "Discovering RoboTwin task folders..."
ROBOTWIN_TASK_PATHS=$(python3 -c "
import os, sys
root = '$ROBOTWIN_DATA_ROOT'
subsets = '$SUBSETS'.split()
paths = []
for subset in subsets:
    subset_dir = os.path.join(root, subset)
    if not os.path.isdir(subset_dir):
        print(f'[WARN] Subset not found: {subset_dir}', file=sys.stderr)
        continue
    for task in sorted(os.listdir(subset_dir)):
        task_dir = os.path.join(subset_dir, task)
        if os.path.isfile(os.path.join(task_dir, 'meta', 'info.json')):
            paths.append(task_dir)
if not paths:
    print('ERROR: No valid task folders found!', file=sys.stderr)
    sys.exit(1)
print(f'Found {len(paths)} task folders', file=sys.stderr)
print('[' + ','.join(paths) + ']')
")

if [ $? -ne 0 ]; then
    echo "ERROR: Failed to discover task paths"
    exit 1
fi

echo "Task paths: ${ROBOTWIN_TASK_PATHS:0:200}..."

# ============ LAUNCH TRAINING ============
torchrun \
    --nnodes=$PET_NNODES \
    --nproc_per_node=$PET_NPROC_PER_NODE \
    --node_rank=$PET_NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --rdzv_conf timeout=3600 \
    groot/vla/experiment/experiment.py \
    report_to=tensorboard \
    data=dreamzero/robotwin \
    wandb_project=dreamzero_robotwin_wan2_2 \
    train_architecture=full \
    num_frames=33 \
    action_horizon=48 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf_wan22 \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=48 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-4 \
    +training_args.max_grad_norm=1.0 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_strategy=steps \
    save_steps=1000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=64 \
    max_steps=10000 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=160 \
    image_resolution_height=128 \
    max_action_dim=16 \
    max_state_dim=16 \
    save_lora_only=false \
    max_chunk_size=4 \
    frame_seqlen=80 \
    action_head_cfg.config.target_video_width=320 \
    action_head_cfg.config.target_video_height=256 \
    robotwin_task_paths="$ROBOTWIN_TASK_PATHS" \
    dit_version=$WAN22_CKPT_DIR \
    text_encoder_pretrained_path=$WAN22_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$IMAGE_ENCODER_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN22_CKPT_DIR/Wan2.2_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR
