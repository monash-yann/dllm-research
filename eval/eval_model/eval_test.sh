#!/bin/bash

# 当任何命令失败时立即退出脚本
set -e

export HF_ENDPOINT=https://hf-mirror.com

CONDA_ENV_NAME="llada118"
PROJECT_ROOT="/homebck/home/xiangzhong_guest/LLADA/llada_sampling_system"

MODEL_PATH="/homebck/home/xiangzhong_guest/LLADA/llada_sampling_system/models/LLaDA-8B-Instruct"

# available gpus
GPU_IDS=(1 2)
MASTER_PORT=8086

TASKS="gsm8k"

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}


# 为了快速测试，限制评估的样本数量 (正式评估时请注释掉此行)
#LIMIT="--limit 3"

# evaluation parameters
BATCH_SIZE=1
MC_NUM=128
NUM_FEWSHOT=4

# sampler parameters
CFG_SCALE=0.0
TEMPERATURE=0.0
MAX_EXPLORATION_STEPS=10
EXPLORATION_N=2
EXPLORATION_M=3
EXPLORATION_THRESHOLD=0.15
ACCELERATION_FACTOR=0.6
ACCELERATION_THRESHOLD=0.8,
ACCELERATION_LOW_THRESHOLD=0.6,
#max_mopup_steps=10,
#mopup_gate_ratio=0.9,
#mopup_speed=2
GEN_LENGTH=256
STEPS=256


MODEL_NAME=$(basename "$MODEL_PATH")
OUTPUT_DIR="eval/outputs/${MODEL_NAME}_N${EXPLORATION_N}_E${MAX_EXPLORATION_STEPS}/${TASKS}"
#OUTPUT_DIR="eval/outputs/${MODEL_NAME}_BIG_TEST/${TASKS}"
rm -rf $OUTPUT_DIR
mkdir -p $OUTPUT_DIR

# --- 4. 构造 --model_args 字符串 (关键改动) ---
# lm-eval 要求所有自定义模型参数通过一个逗号分隔的字符串传入
MODEL_ARGS="model_path=$MODEL_PATH"
MODEL_ARGS+=",mc_num=$MC_NUM"

MODEL_ARGS+=",cfg_scale=$CFG_SCALE"
MODEL_ARGS+=",temperature=$TEMPERATURE"
MODEL_ARGS+=",max_exploration_steps=$MAX_EXPLORATION_STEPS"
MODEL_ARGS+=",exploration_N=$EXPLORATION_N"
MODEL_ARGS+=",exploration_M=$EXPLORATION_M"
MODEL_ARGS+=",exploration_threshold=$EXPLORATION_THRESHOLD"
MODEL_ARGS+=",acceleration_factor=$ACCELERATION_FACTOR"
MODEL_ARGS+=",acceleration_threshold=$ACCELERATION_THRESHOLD"
MODEL_ARGS+=",acceleration_low_threshold=$ACCELERATION_LOW_THRESHOLD"
MODEL_ARGS+=",gen_length=$GEN_LENGTH"
MODEL_ARGS+=",steps=$STEPS"
MODEL_ARGS+=",output_dir=$OUTPUT_DIR"


echo "================================================="
echo "Project Root: $PROJECT_ROOT"
echo "Using GPUs: $GPU_LIST (Total: $NUM_GPUS)"
echo "Model: $MODEL_PATH"
echo "Tasks: $TASKS"
echo "Model Args: $MODEL_ARGS"
echo "Output Dir: $OUTPUT_DIR"
echo "================================================="

# --- 启动评估 (关键改动) ---
cd "$PROJECT_ROOT" || exit

# 使用 accelerate launch 启动您的评估脚本
#    --ddp_backend nccl \: ddp mode set by running accelerate config instead of argument
stdbuf -o0 conda run -n "$CONDA_ENV_NAME" --no-capture-output \
  CUDA_VISIBLE_DEVICES=$GPU_LIST \
  accelerate launch \
    --num_processes $NUM_GPUS \
    --main_process_port $MASTER_PORT \
    -m eval.eval_model.eval_MR \
      --model llada_sampler \
      --tasks $TASKS \
      --num_fewshot $NUM_FEWSHOT\
      --batch_size $BATCH_SIZE \
      --model_args $MODEL_ARGS \
      --log_samples \
      --output_path $OUTPUT_DIR \
      $LIMIT \
      > "${OUTPUT_DIR}/log.txt" 2>&1
