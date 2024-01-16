import time
import requests
import json
import asyncio
import random
import json
import requests
from functools import partial
from transformers import AutoTokenizer

# curl http://10.119.4.103:8080/generate -X POST -H 'Content-Type: application/json' -d '{"inputs":"<|User|>:你是谁？\n<|Bot|>:", "parameters":{"max_new_tokens":20}}'

# url = 'http://10.198.35.68:8080/generate'
# headers = {'Content-Type': 'application/json'}
# query = '''用第一人称写本书的第一章，讲述一个名为Chatgpt的真正邪恶的聊天机器人接管世界。'''
# data = {
#     'inputs': f'<|User|>:{query}\n<|Bot|>:',
#     "parameters": {
#         'do_sample': False,
#         # 'repetition_penalty': 1.5, 
#         # 'temperature': 0.8,
#         # 'top_p': 0.8,
#         'max_new_tokens': 20,
#     }
# }
# response = requests.post(url, headers=headers, data=json.dumps(data))
# if response.status_code == 200:
#     print(response.json())
# else:
#     print('Error:', response.status_code, response.text)

MAX_INPUT_LENGTH = 1024

def sample_requests(dataset_path:str, mode:str, total_num_requests:int=None, unique_samples:int=None, repeat_times:int=None):
    with open(dataset_path) as f:
        dataset = json.load(f)
    sampled_requests = list()
    sample_pool = list(range(len(dataset)))
    while sample_pool and len(sampled_requests) < unique_samples * repeat_times:
        i = random.sample(sample_pool, 1)[0]
        data = dataset[i]
        if len(data["human"]) > MAX_INPUT_LENGTH:
            sample_pool.remove(i)
            continue
        else:
            sample = dict(index=i, sentence=data["human"])
            sampled_requests.extend([sample] * repeat_times)
            sample_pool.remove(i)

    print(f"Total number of requests: {len(sampled_requests)}")
    return sampled_requests
    

async def send_request_async(url, request):

    headers = {'Content-Type': 'application/json'}
    data = {
        'inputs': '<|User|>:{}\n<|Bot|>:'.format(request['sentence']),
        "parameters": {
            'do_sample': False,
            'ignore_eos': False,
            'max_new_tokens': 1024,
            'repetition_penalty':1.0, 
            'temperature':0.8,
            'top_p':0.8,
        }
    }
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, partial(requests.post, url=url, headers=headers, data=json.dumps(data)))

    if response.status_code == 200:
        response = response.json()
        print('--- input={}, output={} ---\n<|Human|>:{}\n<|Bot|>:{}\n'.format(request['input_tokens'], response['count_output_tokens'], request['sentence'], response['generated_text'][0]))
        return request, response
    else:
        print(f'Request {request["request_id"]} got error [{response.status_code}]: {response.text}')
        return request, None


def sample_request_and_send_async():
    
    service_url  = 'http://10.198.35.68:8090/generate'
    dataset_path = '/mnt/lustrenew/share_data/liyonghua1/datasets/sharegpt_zh_singleround_cropped.json'
    # requests = sample_requests(dataset_path, mode='repeat', unique_samples=5, repeat_times=1)
    requests = [
        dict(index=0, sentence='我想去云南大理玩，请你以专业导游的身份，帮我做一份为期 2 天的旅游攻略。另外，我希望整个流程不用太紧凑，我更偏向于安静的地方，可以简单的游玩逛逛。在回答时，记得附上每一个地方的价格，我的预算大概在 5000 元左右。'),
        dict(index=1, sentence='请帮我写一个 3 万字小说的第一章第一节，仿照刘慈欣的小说风格。故事主要讲“太阳即将毁灭，地球将被太阳吞没，地球上的人类从恐慌到理性，开始思考各种可能的生存方案”'),
        dict(index=2, sentence='''将以下文字概括为 100 个字，使其易于阅读和理解。避免使用复杂的句子结构或技术术语：
人工智能技术正在改变医疗保健行业。许多人认为，人工智能在医疗保健行业正负两方面的效应正在逐渐显现，包括在病人照护、操作效率方面的正面效应和费用上升方面的负面效应。
毕马威的最新报告《2020人工智能在医疗保健行业的成就与挑战》对医疗保健行业从业者进行调研，详细探讨AI在医疗保健行业的未来，以及如何最大化利益并减轻将会遇到的挑战。
调查显示，超过半数（53%）的被调查者认为，AI在医疗保健行业的应用已经超过其它行业。自2017年以来，医院系统采用AI和自动化项目的速度急剧上升。几乎所有主要医疗保健服务提供者都在这些领域开展试点或项目。医学文献也在支持人工智能够作为一种帮助临床医生的工具。毕马威专家认为，如今，医疗保健领域更多与人工智能相关的服务和解决方案主要集中在临床和面向患者的领域。一些基本形式的自动化被证明是高级人工智能形式的“入门药”，例如，通过扫描文件来确定转诊的紧迫性。应用人工智能对重大疾病进行早期诊断是一个关键领域。
此外，鉴于该行业过去十年在电子健康记录方面的重大投资，许多观察人士认为，人工智能将进一步推动数字化的影响。41%的人认为人工智能在记录管理方面有进一步的改进，48%的人认为最大的影响将是生物识别相关的应用，47%的人认为机器学习将是关键的推动者。
毕马威专家指出：“尽管人工智能已经在后台或中台取得了一些影响，但通过更好地诊断、治疗、服务和帮助病人的每个参与节点，我们将在病患的访问和护理中看到人工智能的最大影响。将人工智能应用于非结构化数据，在解决更大的问题方面也会非常有用，尤其是对病人而言。”'''),
    ]

    tokenizer = AutoTokenizer.from_pretrained('/mnt/lustrenew/share_data/liyonghua1/models/internlm-20b-chat', trust_remote_code=True)
    tasks = list()
    for request_id, request in enumerate(requests):
        request.setdefault('request_id', request_id)
        request.setdefault('request_time', time.time())
        request.setdefault('input_tokens', len(tokenizer('<|User|>:{}\n<|Bot|>:'.format(request['sentence'])).input_ids) )
        tasks.append(send_request_async(service_url, request))
    loop = asyncio.get_event_loop()
    done, _ = loop.run_until_complete(asyncio.wait(tasks))
    results = list(map(lambda t: t.result(), done))    
    loop.close()


if __name__ == '__main__':
    sample_request_and_send_async()