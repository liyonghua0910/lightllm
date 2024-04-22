#!/bin/bash

source activate /home/mnt/liyonghua1/envs/lightllm

SERVICE_NAME="llama2_7b_h2o_cum_comp20_realloc"  # llama2_lightllm_ori
SERVICE_IP="10.119.48.255"
DATASETS="Xsum humaneval gsm8k"    # [mmlu cmmlu agieval triviaqa Xsum humaneval math gsm8k]
MODE="all"    # [all | infer | eval | viz]
WORK_DIR="./outputs/exp1"
REUSED_DIR="$(date +%Y%m%d_%H%M%S)_${SERVICE_NAME}"
# REUSED_DIR="debug"
# 20240408_190658_llama2_7b_fullcache
# 20240409_224107_llama2_7b_h2o
# 20240410_001534_llama2_7b_hybrid
# 20240410_190802_llama2_7b_sink

cd /home/mnt/liyonghua1/opencompass
formatted_datasets=$(echo $DATASETS | sed -E 's/[^ ]+/*&_datasets,/g')
sed -i "s/url='.*'/url='http:\/\/$SERVICE_IP:8080\/generate'/" configs/eval_lightllm.py
sed -i "s/abbr='.*'/abbr='$SERVICE_NAME'/" configs/eval_lightllm.py
sed -i "s/datasets = \[.*\]/datasets = [$formatted_datasets]/" configs/eval_lightllm.py

if test -z $REUSED_DIR; then
    python run.py configs/eval_lightllm.py --work-dir $WORK_DIR --mode $MODE --debug
else
    python run.py configs/eval_lightllm.py --work-dir $WORK_DIR --mode $MODE --reuse $REUSED_DIR --debug
fi
