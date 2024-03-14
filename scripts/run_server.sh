#!/bin/bash

source activate /home/mnt/liyonghua1/envs/lightllm

LIGHTLLM_DIR=/home/mnt/liyonghua1/lightllm
LOGGING_DIR=$LIGHTLLM_DIR/logs
mkdir -p $LOGGING_DIR
cd $LIGHTLLM_DIR

IP_ADDR=$(hostname -I | awk '{print $1}')
echo "ip address is: $IP_ADDR"
sed -i "s/IP_ADDR=\"[^\"]*\"/IP_ADDR=\"$IP_ADDR\"/" $LIGHTLLM_DIR/scripts/request.sh

LIGHTLLM_LOG_LEVEL=debug \
ENABLE_CACHE_DROPPING=1 \
CUDA_LAUNCH_BLOCKING=0 \
CACHE_SINK_SIZE=4 \
CACHE_TOP_SIZE=30 \
CACHE_LOCAL_SIZE=30 \
python -m lightllm.server.api_server \
    --model_dir /home/mnt/sdc-share/models/Llama-2-7b-hf --trust_remote_code    \
    --host 0.0.0.0 --port 8080 \
    --tp 4 --max_total_token_num 12000 | tee $LOGGING_DIR/`date +%y%m%d_%H%M%S`.log

# /home/mnt/sdc-share/models/internlm-20b-chat