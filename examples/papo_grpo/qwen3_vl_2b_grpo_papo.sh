#!/bin/bash

set -x

CUDA_IDS=0,1,2,3
N_GPU=4

export PYTHONUNBUFFERED=1
export RAY_memory_usage_threshold=0.98

MODEL_PATH=Qwen/Qwen3-VL-2B-Thinking

TOTAL_EPOCHES=2
GLOBAL_BATCH_SIZE=128
ROLLOUT_BATCH_SIZE=384
VAL_BATCH_SIZE=512
MAX_PROMPT_LENGTH=4096

EXP_NAME="qwen3_vl_2b__thinking__grpo__papo__fix_reward__REC_ENV_EXP_kl_prcp_0.01__ep${TOTAL_EPOCHES}_rb${ROLLOUT_BATCH_SIZE}_gb${GLOBAL_BATCH_SIZE}"

CONGI_FILE="examples/configs/config_grpo_papo.yaml"
TRAIN_FILE="PAPOGalaxy/PAPO_ViRL39K_train"
VAL_FILE="PAPOGalaxy/PAPO_MMK12_test"

FORMAT_PROMPT="examples/format_prompt/math_perception.jinja"
REWARD_FUNCTION="examples/reward_function/qwen3_vl_think.py:compute_score"

KL_PRCP_COEF=0.01

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
    algorithm.kl_prcp_coef=${KL_PRCP_COEF} \
    trainer.val_freq=-1

