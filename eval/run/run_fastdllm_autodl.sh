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

#N_LIMIT=4

#TASKS="gsm8k"
#NUM_FEWSHOT=4

#TASKS="mbpp"

#TASKS="humaneval"

# math-500 is a dataset on huggingface
#TASKS="math-500"
#INCLUDE_PATH="$PROJECT_ROOT/eval/tasks/math-500/"

TASKS=sudoku
INCLUDE_PATH="$PROJECT_ROOT/eval/tasks/sudoku/"
NUM_FEWSHOT=4

GPU_LIST=$(IFS=,; echo "${GPU_IDS[*]}")
NUM_GPUS=${#GPU_IDS[@]}


# evaluation parameters
BATCH_SIZE=1
MC_NUM=128
# sampler parameters
CFG_SCALE=0.0
TEMPERATURE=0.0
POSITIONAL_WEIGHTS_TYPE='none'
MAX_WEIGHT=1.0
INITIAL_MIN_WEIGHT=0.0
REMASKING="low_confidence"
DECODING_METHOD="topk"
FACTOR=1.0
CONFIDENCE_THRESHOLD=0.9
K=1

MODEL_NAME=$(basename "$MODEL_PATH")

SL_VALUES=(128)

BLOCK_LENGTH=128

for SL in "${SL_VALUES[@]}"
do
  echo "========================== evaluating SL=${SL}, BL=${BL} =========================="

  GEN_LENGTH=$SL
  STEPS=$SL

  if [ "$DECODING_METHOD" = "fixed" ]; then
    METHOD_SUFFIX="conf_tr${CONFIDENCE_THRESHOLD}"
  elif [ "$DECODING_METHOD" = "factor" ]; then
    METHOD_SUFFIX="factor${FACTOR}"
  elif [ "$DECODING_METHOD" = "topk" ]; then
    METHOD_SUFFIX="k${K}"
  else
    METHOD_SUFFIX=""
  fi
  OUTPUT_DIR="eval/outputs/${MODEL_NAME}_pure_MTD${DECODING_METHOD}_PWT${POSITIONAL_WEIGHTS_TYPE}_imw${INITIAL_MIN_WEIGHT}_${N_LIMIT:+limit_$N_LIMIT}/${TASKS}/SL${SL}_BL${BLOCK_LENGTH}/${METHOD_SUFFIX}"
  rm -rf $OUTPUT_DIR
  mkdir -p $OUTPUT_DIR

  # lm-eval 要求所有自定义模型参数通过一个逗号分隔的字符串传入
  MODEL_ARGS="model_path=$MODEL_PATH"
  MODEL_ARGS+=",output_dir=$OUTPUT_DIR"
  MODEL_ARGS+=",mc_num=$MC_NUM"
  MODEL_ARGS+=",gen_length=$GEN_LENGTH"
  MODEL_ARGS+=",steps=$STEPS"
  MODEL_ARGS+=",block_length=$BLOCK_LENGTH"

  MODEL_ARGS+=",cfg_scale=$CFG_SCALE"
  MODEL_ARGS+=",temperature=$TEMPERATURE"
  MODEL_ARGS+=",positional_weights_type=$POSITIONAL_WEIGHTS_TYPE"
  MODEL_ARGS+=",max_weight=$MAX_WEIGHT"
  MODEL_ARGS+=",initial_min_weight=$INITIAL_MIN_WEIGHT"
  MODEL_ARGS+=",block_length=$BLOCK_LENGTH"
  MODEL_ARGS+=",remasking=$REMASKING"
  MODEL_ARGS+=",decoding_method=$DECODING_METHOD"
  MODEL_ARGS+=",factor=$FACTOR"
  MODEL_ARGS+=",confidence_threshold=$CONFIDENCE_THRESHOLD"
  MODEL_ARGS+=",k=$K"

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
  stdbuf -o0 "$CONDA_EXE" run -n "$CONDA_ENV_NAME" --no-capture-output \
    CUDA_VISIBLE_DEVICES=$GPU_LIST \
    accelerate launch \
      --num_processes $NUM_GPUS \
      --main_process_port $MASTER_PORT \
      -m eval.eval_model.eval_pure_llada \
        --model eval_sampler \
        --confirm_run_unsafe_code \
        --tasks $TASKS \
        ${INCLUDE_PATH:+--include_path $INCLUDE_PATH} \
        ${NUM_FEWSHOT:+--num_fewshot $NUM_FEWSHOT}\
        --batch_size $BATCH_SIZE \
        --model_args $MODEL_ARGS \
        --log_samples \
        --output_path $OUTPUT_DIR \
        ${N_LIMIT:+--limit $N_LIMIT} \
        > "${OUTPUT_DIR}/log.txt" 2>&1
done
# only in autodl
/usr/bin/shutdown
