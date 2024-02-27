#!/bin/bash

TEMPFILE=run_opencompass_`date +%Y%m%d%H%M%S`.sh
cp run_opencompass.sh temp/$TEMPFILE

srun -p fe5a2809-3094-4c11-935d-be8a5acb81a7 --workspace-id bbfd49c7-bffd-4212-9288-2f85bdab54de -r N1lS.Ib.I20.1 \
    --container-image registry.st-sh-01.sensecore.cn/scg_sdc_ccr/aicl-c1595115-adf0-11ee:20240115-15h10m57s \
    -j opencompass -f pytorch bash /home/mnt/liyonghua1/lightllm/scripts/temp/$TEMPFILE