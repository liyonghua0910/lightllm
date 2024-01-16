#!/bin/bash

WORKING_DIR=/home/mnt/liyonghua1/lightllm
LOGGING_DIR=$WORKING_DIR/logs
mkdir -p $LOGGING_DIR
source activate /home/mnt/liyonghua1/envs/lightllm
cd $WORKING_DIR

IP_ADDR=$(hostname -I | awk '{print $1}')
echo "ip address is: $IP_ADDR"
sed -i "s/IP_ADDR=\"[^\"]*\"/IP_ADDR=\"$IP_ADDR\"/" request.sh

LIGHTLLM_DEBUG=0 \
ENABLE_HEAVY_HITTER_ORACLE=1 \
CUDA_LAUNCH_BLOCKING=0 \
REQUEST_CACHE_SIZE=32 \
REQUEST_CACHE_SPLIT=0.5 \
python -m lightllm.server.api_server \
    --model_dir /home/mnt/sdc-share/models/internlm-20b-chat --trust_remote_code    \
    --host 0.0.0.0 --port 8080 \
    --tp 2 --max_total_token_num 12000 | tee $LOGGING_DIR/`date +%y%m%d_%H%M%S`.log