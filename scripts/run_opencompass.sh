#!/bin/bash

source activate /home/mnt/liyonghua1/envs/lightllm

SERVICE_NAME="sink4_top8_local8" # [full_cache | top64_local64 | sink4_top64_local64 | top32_local32 | top16_local16 | top8_local8 | sink4_top8_local8 | top4_local4]
DATASETS="Xsum" # [mmlu cmmlu agieval triviaqa Xsum humaneval math gsm8k]
MODE="all" # [all | infer | eval | viz]
WORK_DIR="./outputs/myworkspace"

if [ "$SERVICE_NAME" == "full_cache" ]; then
    SERVICE_IP="10.119.37.46"
    REUSED_DIR="20240118_005347_internlm20b_fullcache"
elif [ "$SERVICE_NAME" == "sink4_top64_local64" ]; then
    SERVICE_IP="10.119.13.87"
    REUSED_DIR="20240226_182350_internlm20b_sink4top64local64"
elif [ "$SERVICE_NAME" == "top64_local64" ]; then
    SERVICE_IP="10.119.17.165"
    REUSED_DIR="20240118_103914_internlm20b_top64local64"
elif [ "$SERVICE_NAME" == "sink4_top8_local8" ]; then
    SERVICE_IP="10.119.4.201"
    REUSED_DIR=""
elif [ "$SERVICE_NAME" == "top8_local8" ]; then
    SERVICE_IP="10.119.17.70"
    REUSED_DIR="20240129_175629_internlm20b_top8local8"
elif [ "$SERVICE_NAME" == "top4_local4" ]; then
    SERVICE_IP="10.119.52.240"
    REUSED_DIR="20240130_155152_internlm20b_top4local4"
fi

cd /home/mnt/liyonghua1/opencompass
formatted_datasets=$(echo $DATASETS | sed -E 's/[^ ]+/*&_datasets,/g')
sed -i "s/url='.*'/url='http:\/\/$SERVICE_IP:8080\/generate'/" configs/eval_lightllm.py
sed -i "s/datasets = \[.*\]/datasets = [$formatted_datasets]/" configs/eval_lightllm.py
python run.py configs/eval_lightllm.py --work-dir $WORK_DIR --reuse $REUSED_DIR --mode $MODE --debug
