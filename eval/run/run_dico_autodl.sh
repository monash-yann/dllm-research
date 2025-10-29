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
GPU_IDS=(0 1)
MASTER_PORT=8086

#N_LIMIT=48

#TASKS="gsm8k"
#NUM_FEWSHOT=4

#TASKS="mbpp"

TASKS="humaneval"

#TASKS="math-500"
#INCLUDE_PATH="$PROJECT_ROOT/eval/tasks/math-500/"


GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}

# evaluation parameters
BATCH_SIZE=1
MC_NUM=128

# sampler parameters
CFG_SCALE=0.0
TEMPERATURE=0.0
MAX_EXPLORATION_STEPS=5
EXPLORATION_N_VALUES=(8)
#EXPLORATION_N_VALUES=(8)
#EXPLORATION_N_VALUES=(7)
EXPLORATION_M=2
EXPLORATION_THRESHOLD=0.3
ACCELERATION_PARALLEL_METHOD='factor'
ACCELERATION_THRESHOLD=0.9
ACCELERATION_LOW_THRESHOLD=0.6
ACCELERATION_FACTOR=1
MOPUP_GATE_RATIO=0.8
MOPUP_MARGIN_THRESHOLD=3
MAX_MOPUP_STEPS=20
MOPUP_SPEED=1

POSITIONAL_WEIGHTS_TYPE='ratio'
MAX_WEIGHT=1.0
INITIAL_MIN_WEIGHT=0.05
UR_FACTOR=0.5

MODEL_NAME=$(basename "$MODEL_PATH")

SL_VALUES=(256)

BLOCK_LENGTHES=(64)

for SL in "${SL_VALUES[@]}"
do
  GEN_LENGTH=$SL
  STEPS=$SL
  for BL in "${BLOCK_LENGTHES[@]}"
  do
    echo "========================== evaluating SL=${SL}, BL=${BL} =========================="
    for EXPLORATION_N in "${EXPLORATION_N_VALUES[@]}"
    do
      echo "========================== evaluating N=${EXPLORATION_N} =========================="
      OUTPUT_DIR="eval/outputs/${MODEL_NAME}_dicodyna+mpmargin_APM${ACCELERATION_PARALLEL_METHOD}_PWT${POSITIONAL_WEIGHTS_TYPE}_DVDonly_imw${INITIAL_MIN_WEIGHT}_ur${UR_FACTOR}_${N_LIMIT:+limit_$N_LIMIT}/${TASKS}/SL${SL}_BL${BL}/N${EXPLORATION_N}E${MAX_EXPLORATION_STEPS}_gate${MOPUP_GATE_RATIO}_factor${ACCELERATION_FACTOR}_exptr${EXPLORATION_THRESHOLD}_acctr${ACCELERATION_THRESHOLD}_mptr${MOPUP_MARGIN_THRESHOLD}"
      rm -rf $OUTPUT_DIR
      mkdir -p $OUTPUT_DIR

      MODEL_ARGS="model_path=$MODEL_PATH"
      MODEL_ARGS+=",mc_num=$MC_NUM"
      MODEL_ARGS+=",gen_length=$GEN_LENGTH"
      MODEL_ARGS+=",steps=$STEPS"
      MODEL_ARGS+=",block_length=$BL"
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
      MODEL_ARGS+=",mopup_margin_threshold=$MOPUP_MARGIN_THRESHOLD"
      MODEL_ARGS+=",max_mopup_steps=$MAX_MOPUP_STEPS"
      MODEL_ARGS+=",mopup_speed=$MOPUP_SPEED"
      MODEL_ARGS+=",positional_weights_type=$POSITIONAL_WEIGHTS_TYPE"
      MODEL_ARGS+=",max_weight=$MAX_WEIGHT"
      MODEL_ARGS+=",initial_min_weight=$INITIAL_MIN_WEIGHT"
      MODEL_ARGS+=",ur_factor=$UR_FACTOR"

      echo "================================================="
      echo "Project Root: $PROJECT_ROOT"
      echo "Using GPUs: $GPU_LIST (Total: $NUM_GPUS)"
      echo "Model: $MODEL_PATH"
      echo "Tasks: $TASKS"
      echo "Model Args: $MODEL_ARGS"
      echo "Output Dir: $OUTPUT_DIR"
      echo "================================================="

      cd "$PROJECT_ROOT" || exit

      #    --ddp_backend nccl \: ddp mode set by running accelerate config instead of argument
      stdbuf -o0 "$CONDA_EXE" run -n "$CONDA_ENV_NAME" --no-capture-output \
        CUDA_VISIBLE_DEVICES=$GPU_LIST \
        accelerate launch \
          --num_processes $NUM_GPUS \
          --main_process_port $MASTER_PORT \
          -m eval.eval_model.eval_MR \
            --model eval_sampler \
            --confirm_run_unsafe_code \
            --tasks $TASKS \
            ${INCLUDE_PATH:+--include_path $INCLUDE_PATH} \
            ${NUM_FEWSHOT:+ --num_fewshot $NUM_FEWSHOT} \
            --batch_size $BATCH_SIZE \
            --model_args $MODEL_ARGS \
            --log_samples \
            --output_path $OUTPUT_DIR \
            ${N_LIMIT:+--limit $N_LIMIT} \
            > "${OUTPUT_DIR}/log.txt" 2>&1
    done
  done
done
# only in autodl
/usr/bin/shutdown
shutdown
