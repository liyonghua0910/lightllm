#!/bin/bash

IP_ADDR="10.119.41.23"  # 由服务启动脚本检测到 pod ip 地址后自动填入

# curl http://$IP_ADDR:8080/generate -X POST -H 'Content-Type: application/json' \
#     -d '{"inputs":"<|User|>:你是谁?\n<|Bot|>:", "parameters":{"max_new_tokens":20}}' &
# curl http://$IP_ADDR:8080/generate -X POST -H 'Content-Type: application/json' -w '\n' \
#     -d '{"inputs":"<|User|>:Document：The ex-Reading defender denied fraudulent trading charges relating to the Sodje Sports Foundation - a charity to raise money for Nigerian sport.\nMr Sodje, 37, is jointly charged with elder brothers Efe, 44, Bright, 50 and Stephen, 42.\nAppearing at the Old Bailey earlier, all four denied the offence.\nThe charge relates to offences which allegedly took place between 2008 and 2014.\nSam, from Kent, Efe and Bright, of Greater Manchester, and Stephen, from Bexley, are due to stand trial in July.\nThey were all released on bail.\nBased on the previous text, provide a brief single summary:\n<|Bot|>:", "parameters":{"max_new_tokens":10}}' &
# curl http://$IP_ADDR:8080/generate -X POST -H 'Content-Type: application/json' -w '\n' \
#     -d '{"inputs":"<|User|>:将以下文字概括为 100 个字，使其易于阅读和理解。避免使用复杂的句子结构或技术术语：\n人工智能技术正在改变医疗保健行业。许多人认为，人工智能在医疗保健行业正负两方面的效应正在逐渐显现，包括在病人照护、操作效率方面的正面效应和费用上升方面的负面效应。\n毕马威的最新报告《2020人工智能在医疗保健行业的成就与挑战》对医疗保健行业从业者进行调研，详细探讨AI在医疗保健行业的未来，以及如何最大化利益并减轻将会遇到的挑战。\n调查显示，超过半数（53%）的被调查者认为，AI在医疗保健行业的应用已经超过其它行业。自2017年以来，医院系统采用AI和自动化项目的速度急剧上升。几乎所有主要医疗保健服务提供者都在这些领域开展试点或项目。医学文献也在支持人工智能够作为一种帮助临床医生的工具。毕马威专家认为，如今，医疗保健领域更多与人工智能相关的服务和解决方案主要集中在临床和面向患者的领域。一些基本形式的自动化被证明是高级人工智能形式的“入门药”，例如，通过扫描文件来确定转诊的紧迫性。应用人工智能对重大疾病进行早期诊断是一个关键领域。\n此外，鉴于该行业过去十年在电子健康记录方面的重大投资，许多观察人士认为，人工智能将进一步推动数字化的影响。41%的人认为人工智能在记录管理方面有进一步的改进，48%的人认为最大的影响将是生物识别相关的应用，47%的人认为机器学习将是关键的推动者。\n毕马威专家指出：“尽管人工智能已经在后台或中台取得了一些影响，但通过更好地诊断、治疗、服务和帮助病人的每个参与节点，我们将在病患的访问和护理中看到人工智能的最大影响。将人工智能应用于非结构化数据，在解决更大的问题方面也会非常有用，尤其是对病人而言。”\n<|Bot|>:", "parameters":{"max_new_tokens":10}}' &
# curl http://$IP_ADDR:8080/generate -X POST -H 'Content-Type: application/json' \
#     -d '{"inputs":"<|User|>:我想去云南大理玩，请你以专业导游的身份，帮我做一份为期 2 天的旅游攻略。另外，我希望整个流程不用太紧凑，我更偏向于安静的地方，可以简单的游玩逛逛。在回答时，记得附上每一个地方的价格，我的预算大概在 5000 元左右。\n<|Bot|>:", "parameters":{"max_new_tokens":10}}' &

# curl http://$IP_ADDR:8080/generate -X POST -H 'Content-Type: application/json' -w '\n' \
#     -d '{"inputs":"This is a brand new world, I have to say. The trees, the clouds, and those cute little ", "parameters":{"max_new_tokens":10}}' &
# curl http://$IP_ADDR:8080/generate -X POST -H 'Content-Type: application/json' -w '\n' \
#     -d '{"inputs":"Now I am doing this debugging test to see whether these two queries made in a batch would affect the computation accuracy. Wait till ", "parameters":{"max_new_tokens":20}}' &
# curl http://$IP_ADDR:8080/generate -X POST -H 'Content-Type: application/json' -w '\n' \
#     -d '{"inputs":"A good habit is", "parameters":{"max_new_tokens":30}}' &
# curl http://$IP_ADDR:8080/generate -X POST -H 'Content-Type: application/json' -w '\n' \
#     -d '{"inputs":"I am", "parameters":{"max_new_tokens":40}}' &
curl http://$IP_ADDR:8080/generate -X POST -H 'Content-Type: application/json' -w '\n' \
    -d '{"inputs":"Document：The ex-Reading defender denied fraudulent trading charges relating to the Sodje Sports Foundation - a charity to raise money for Nigerian sport.\nMr Sodje, 37, is jointly charged with elder brothers Efe, 44, Bright, 50 and Stephen, 42.\nAppearing at the Old Bailey earlier, all four denied the offence.\nThe charge relates to offences which allegedly took place between 2008 and 2014.\nSam, from Kent, Efe and Bright, of Greater Manchester, and Stephen, from Bexley, are due to stand trial in July.\nThey were all released on bail.\nBased on the previous text, provide a brief single summary:", "parameters":{"max_new_tokens":20}}' &
curl http://$IP_ADDR:8080/generate -X POST -H 'Content-Type: application/json' -w '\n' \
    -d '{"inputs":"Document：The ex-Reading defender denied fraudulent trading charges relating to the Sodje Sports Foundation - a charity to raise money for Nigerian sport.\nMr Sodje, 37, is jointly charged with elder brothers Efe, 44, Bright, 50 and Stephen, 42.\nAppearing at the Old Bailey earlier, all four denied the offence.\nThe charge relates to offences which allegedly took place between 2008 and 2014.\nSam, from Kent, Efe and Bright, of Greater Manchester, and Stephen, from Bexley, are due to stand trial in July.\nThey were all released on ", "parameters":{"max_new_tokens":40}}' &
curl http://$IP_ADDR:8080/generate -X POST -H 'Content-Type: application/json' -w '\n' \
    -d '{"inputs":"Document：The ex-Reading defender denied fraudulent trading charges relating to the Sodje Sports Foundation - a charity to raise money for Nigerian sport.\nMr Sodje, 37, is jointly charged with elder brothers Efe, 44, Bright, 50 and Stephen, 42.\nAppearing at the Old Bailey earlier, all four denied the offence.\nThe charge", "parameters":{"max_new_tokens":20}}' &
curl http://$IP_ADDR:8080/generate -X POST -H 'Content-Type: application/json' -w '\n' \
    -d '{"inputs":"Document：The ex-Reading defender ", "parameters":{"max_new_tokens":30}}' &
