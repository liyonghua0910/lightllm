#!/bin/bash
# 1.0 => 10.119.17.33
# 0.0 => 10.119.50.86
IP_ADDR="10.119.2.167"  # 由服务启动脚本检测到 pod ip 地址后自动填入

curl http://$IP_ADDR:8080/generate -X POST -H 'Content-Type: application/json' \
    -d '{"inputs":"<|User|>:你是谁?\n<|Bot|>:", "parameters":{"max_new_tokens":100}}'

# curl http://$IP_ADDR:8080/generate -X POST -H 'Content-Type: application/json' \
#     -d '{"inputs":"<|User|>:我想去云南大理玩，请你以专业导游的身份，帮我做一份为期 2 天的旅游攻略。另外，我希望整个流程不用太紧凑，我更偏向于安静的地方，可以简单的游玩逛逛。在回答时，记得附上每一个地方的价格，我的预算大概在 5000 元左右。\n<|Bot|>:", "parameters":{"max_new_tokens":100}}'

echo ""