#!/bin/bash
set -x

CUDA_IDS=0,1
N_GPU=2

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct

TOTAL_EPOCHES=2
GLOBAL_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=384
VAL_BATCH_SIZE=512
MAX_PROMPT_LENGTH=4096

EXP_NAME="qwen2_5_vl_3b__papo_high__ep${TOTAL_EPOCHES}_rb${ROLLOUT_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}"

CONGI_FILE="examples/configs/config_papo.yaml"
TRAIN_FILE="PAPOGalaxy/PAPO_train"
VAL_FILE="PAPOGalaxy/PAPO_val"

FORMAT_PROMPT="examples/format_prompt/math_format_perception.jinja"
REWARD_FUNCTION="examples/reward_function/math.py:compute_score"

KL_PRCP_COEF=0.02

export RAY_memory_usage_threshold=0.98
CUDA_VISIBLE_DEVICES=${CUDA_IDS} python3 -m verl.trainer.main \
    config=${CONGI_FILE} \
    data.train_files=${TRAIN_FILE} \
    data.val_files=${VAL_FILE} \
    data.rollout_batch_size=${ROLLOUT_BATCH_SIZE} \
    data.format_prompt=${FORMAT_PROMPT} \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    worker.actor.global_batch_size=${GLOBAL_BATCH_SIZE} \
    trainer.experiment_name=${EXP_NAME} \
    trainer.n_gpus_per_node=${N_GPU} \
    trainer.total_epochs=${TOTAL_EPOCHES} \
    worker.reward.reward_function=${REWARD_FUNCTION} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    algorithm.kl_prcp_coef=${KL_PRCP_COEF}
