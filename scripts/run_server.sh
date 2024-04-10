#!/bin/bash

source activate /home/mnt/liyonghua1/envs/lightllm

LIGHTLLM_DIR=/home/mnt/liyonghua1/lightllm
LOGGING_DIR=$LIGHTLLM_DIR/logs
mkdir -p $LOGGING_DIR
cd $LIGHTLLM_DIR

IP_ADDR=$(hostname -I | awk '{print $1}')
echo "ip address is: $IP_ADDR"
sed -i "s/IP_ADDR=\"[^\"]*\"/IP_ADDR=\"$IP_ADDR\"/" $LIGHTLLM_DIR/scripts/request.sh
sed -i "s/SERVICE_IP=\"[^\"]*\"/SERVICE_IP=\"$IP_ADDR\"/" $LIGHTLLM_DIR/scripts/run_opencompass.sh

ENABLE_DEBUGPY=1 \
LIGHTLLM_LOG_LEVEL=debug \
CUDA_LAUNCH_BLOCKING=0 \
ENABLE_CACHE_DROPPING=1 \
MIN_CACHE_SIZE=64 \
MAX_CACHE_SIZE=1024 \
COMPRESSION_RATE=0.2 \
CACHE_RECIPE=Hybrid \
python -m lightllm.server.api_server \
    --model_dir /home/mnt/sdc-share/models/Llama-2-7b-hf --trust_remote_code    \
    --host 0.0.0.0 --port 8080 --max_req_input_len 3072 --max_req_total_len 4096 \
    --tp 1 --max_total_token_num 8000 | tee $LOGGING_DIR/`date +%y%m%d_%H%M%S`.log

# /home/mnt/sdc-share/models/internlm-20b-chat