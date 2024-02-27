#!/bin/bash

source activate /home/mnt/liyonghua1/envs/lightllm

now=`date +%Y%m%d_%H%M%S`
n_shots=5
prompt_format=standard
max_request_num=-1
description=${prompt_format}_req${max_request_num}_shot${n_shots}_${now}
cd /home/mnt/liyonghua1/lightllm/scripts
python benchmark.py \
    --data_dir /home/mnt/liyonghua1/datasets/mmlu \
    --save_dir /home/mnt/liyonghua1/lightllm/scripts/results \
    --service_url http://10.119.1.225:8080/generate \
    --max_request_num $max_request_num \
    --n_shots $n_shots \
    --prompt_format $prompt_format \
    --description $description | tee results/$description.log