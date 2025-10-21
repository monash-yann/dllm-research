#!/bin/bash

# 当任何命令失败时立即退出脚本
set -e
export CONDA_EXE="/root/miniconda3/bin/conda"
export HF_ENDPOINT=https://hf-mirror.com
export HF_ALLOW_CODE_EVAL=1

CONDA_ENV_NAME="dico"
PROJECT_ROOT="/root/autodl-tmp/dllm_sampling_system"
MODEL_PATH="$PROJECT_ROOT/models/LLaDA-8B-Instruct"

# available gpus
GPU_IDS=(0 1 2 3)
MASTER_PORT=8086

#TASKS="gsm8k"
#NUM_FEWSHOT=4
#N_LIMIT=6

#TASKS="mbpp"

TASKS="humaneval"

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}

# evaluation parameters
BATCH_SIZE=1
MC_NUM=128
#NUM_FEWSHOT=
# sampler parameters
CFG_SCALE=0.0
TEMPERATURE=0.0
MAX_EXPLORATION_STEPS=10
EXPLORATION_N_VALUES=(1 2 4 8)
#EXPLORATION_N_VALUES=(8 7 6 5 4 3 2)
EXPLORATION_M=2
EXPLORATION_THRESHOLD=0.25
ACCELERATION_PARALLEL_METHOD='fixed'
ACCELERATION_FACTOR=1
ACCELERATION_THRESHOLD=0.9
ACCELERATION_LOW_THRESHOLD=0.6
POSITIONAL_WEIGHTS_TYPE='ratio'
MOPUP_GATE_RATIO=0.8
MAX_MOPUP_STEPS=50
MOPUP_SPEED=2

MAX_WEIGHT=1.0
INITIAL_MIN_WEIGHT=0.0

MODEL_NAME=$(basename "$MODEL_PATH")

SL_VALUES=(256)

for SL in "${SL_VALUES[@]}"
do
  echo "========================== evaluating SL=${SL} =========================="
  GEN_LENGTH=$SL
  STEPS=$SL
  BLOCK_LENGTH=$SL

  for EXPLORATION_N in "${EXPLORATION_N_VALUES[@]}"
  do
    echo "========================== evaluating N=${EXPLORATION_N} =========================="
    OUTPUT_DIR="eval/outputs/${MODEL_NAME}_dico_APM${ACCELERATION_PARALLEL_METHOD}_PWT${POSITIONAL_WEIGHTS_TYPE}_DVDonly_imw${INITIAL_MIN_WEIGHT}_${N_LIMIT:+limit_$N_LIMIT}/${TASKS}/SL${STEPS}/N${EXPLORATION_N}"
    rm -rf $OUTPUT_DIR
    mkdir -p $OUTPUT_DIR

    # --- 构造 --model_args 字符串 (关键改动) ---
    # lm-eval 要求所有自定义模型参数通过一个逗号分隔的字符串传入
    MODEL_ARGS="model_path=$MODEL_PATH"
    MODEL_ARGS+=",mc_num=$MC_NUM"
    MODEL_ARGS+=",gen_length=$GEN_LENGTH"
    MODEL_ARGS+=",steps=$STEPS"
    MODEL_ARGS+=",block_length=$BLOCK_LENGTH"
    MODEL_ARGS+=",output_dir=$OUTPUT_DIR"

    MODEL_ARGS+=",cfg_scale=$CFG_SCALE"
    MODEL_ARGS+=",temperature=$TEMPERATURE"
    MODEL_ARGS+=",max_exploration_steps=$MAX_EXPLORATION_STEPS"
    MODEL_ARGS+=",exploration_N=$EXPLORATION_N"
    MODEL_ARGS+=",exploration_M=$EXPLORATION_M"
    MODEL_ARGS+=",exploration_threshold=$EXPLORATION_THRESHOLD"
    MODEL_ARGS+=",acceleration_parallel_method=$ACCELERATION_PARALLEL_METHOD"
    MODEL_ARGS+=",acceleration_factor=$ACCELERATION_FACTOR"
    MODEL_ARGS+=",acceleration_threshold=$ACCELERATION_THRESHOLD"
    MODEL_ARGS+=",acceleration_low_threshold=$ACCELERATION_LOW_THRESHOLD"
    MODEL_ARGS+=",mopup_gate_ratio=$MOPUP_GATE_RATIO"
    MODEL_ARGS+=",max_mopup_steps=$MAX_MOPUP_STEPS"
    MODEL_ARGS+=",mopup_speed=$MOPUP_SPEED"
    MODEL_ARGS+=",positional_weights_type=$POSITIONAL_WEIGHTS_TYPE"
    MODEL_ARGS+=",max_weight=$MAX_WEIGHT"
    MODEL_ARGS+=",initial_min_weight=$INITIAL_MIN_WEIGHT"


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

    #    --ddp_backend nccl \: ddp mode set by running accelerate config instead of argument
    stdbuf -o0 conda run -n "$CONDA_ENV_NAME" --no-capture-output \
      CUDA_VISIBLE_DEVICES=$GPU_LIST \
      accelerate launch \
        --num_processes $NUM_GPUS \
        --main_process_port $MASTER_PORT \
        -m eval.eval_model.eval_MR \
          --model eval_sampler \
          --confirm_run_unsafe_code \
          --tasks $TASKS \
          ${NUM_FEWSHOT:+ --num_fewshot $NUM_FEWSHOT} \
          --batch_size $BATCH_SIZE \
          --model_args $MODEL_ARGS \
          --log_samples \
          --output_path $OUTPUT_DIR \
          ${N_LIMIT:+--limit $N_LIMIT} \
          > "${OUTPUT_DIR}/log.txt" 2>&1

  done

done
# only in autodl
#/usr/bin/shutdown
