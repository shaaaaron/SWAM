export ROBOTWIN2_DATA_ROOT=${ROBOTWIN2_DATA_ROOT:?Set ROBOTWIN2_DATA_ROOT}
export WAN22_CKPT_DIR=${WAN22_CKPT_DIR:?Set WAN22_CKPT_DIR}
export IMAGE_ENCODER_DIR=${IMAGE_ENCODER_DIR:?Set IMAGE_ENCODER_DIR}
export TOKENIZER_DIR=${TOKENIZER_DIR:?Set TOKENIZER_DIR}
export NUM_GPUS=8
export OUTPUT_DIR=${OUTPUT_DIR:-./checkpoints/dreamzero_robotwin2_subset_adjust_bottle}

# 数据子集选择：
# - 空字符串 "" 表示使用 ROBOTWIN2_DATA_ROOT 下所有 LeRobot 数据文件夹
# - 多个 token 用空格分隔；文件夹名包含任意一个 token 就会被选中
# - 例如只训一个任务：export SUBSETS="beat_block_hammer"
# - 例如只训 clean 数据：export SUBSETS="aloha-agilex_clean_50"
export SUBSETS="adjust_bottle"

# shard 内 step 采样比例。0.1 表示每个 shard 只取约 10% step。
export DATASET_SHARD_SAMPLING_RATE=0.8

# 小数据集建议缩短训练和更频繁保存；需要跑长一点时再调大。
export MAX_STEPS=4000
export SAVE_STEPS=1000
export TRAIN_LR=1e-5
export WARMUP_RATIO=0.03
export PER_DEVICE_TRAIN_BATCH_SIZE=32

bash scripts/train/robotwin2_training_wan22_lr1e_4.sh
