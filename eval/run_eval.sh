#!/bin/bash

# 当任何命令失败时立即退出脚本
set -e

export HF_ENDPOINT=https://hf-mirror.com

CONDA_ENV_NAME="llada118"
PROJECT_ROOT="/homebck/home/xiangzhong_guest/LLADA/llada_sampling_system"

# 用于评估的模型路径
model_path="/homebck/home/xiangzhong_guest/LLADA/llada_sampling_system/models/LLaDA-8B-Instruct"
# 【改动 1】: 从模型路径自动获取模型名称
model_name=$(basename "$model_path")

GPU_IDS=(1 2)
MASTER_PORT=8086
 
# 评估任务和生成长度的组合，与.py中哈希表的key对应
#TASKS=("gsm8k" "math" "countdown" "svamp")
TASKS=("gsm8k")
#GEN_LENGTHS=(128 256 512)
GEN_LENGTHS=(256)
token_per_step=1
temperature=0.0
OUTPUT_SUFFIX='test'

# 如果从命令行提供了 GPU ID，则覆盖默认值
if [ $# -gt 0 ]; then
  GPU_IDS=()
  for arg in "$@"; do
    GPU_IDS+=("$arg")
  done
fi

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}
echo "used GPU: $GPU_LIST (nproc_per_node=$NUM_GPUS)"

# --- do evaluation ---
# go to the root directory of the project
cd "$PROJECT_ROOT" || exit
echo "current project's root directory: $(pwd)"

for task in "${TASKS[@]}"; do
  for gen_length in "${GEN_LENGTHS[@]}"; do

    batch_size=1

    diffusion_steps=$((gen_length / token_per_step))

    echo "------------------------------------------------------------"
    echo "evaluate: $task, gen_length: $gen_length, Batch Size: $batch_size"
    echo "------------------------------------------------------------"

    OUTPUT_DIR="eval/outputs/${model_name}/${task}_gen_${gen_length}_steps_${diffusion_steps}_temp_${temperature}${OUTPUT_SUFFIX}"
#    mkdir -p "$OUTPUT_DIR"

    conda run -n "$CONDA_ENV_NAME" --no-capture-output \
      CUDA_VISIBLE_DEVICES=$GPU_LIST torchrun \
        --nproc_per_node $NUM_GPUS \
        --master_port $MASTER_PORT \
        -m eval.eval \
        --dataset $task \
        --batch_size $batch_size \
        --gen_length $gen_length \
        --diffusion_steps $diffusion_steps \
        --temperature $temperature \
        --output_dir $OUTPUT_DIR \
        --model_path $model_path \
        --model_name $model_name \
        --few_shot 4 \

  done
done

echo "all tasks have been evaluated!"