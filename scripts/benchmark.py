import argparse
from collections import defaultdict
import os
import time
import requests
import json
import asyncio
import random
import logging
import pandas
import json
import requests
from functools import partial
from transformers import AutoTokenizer

random.seed(40)

def build_standard_prompt(subject:str, sample:pandas.DataFrame, shots:pandas.DataFrame):
    ''' 原始的 MMLU 测评提示词构造方式 
        https://github.com/hendrycks/test/blob/master/evaluate.py 
    '''
    n_shots = len(shots)
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(subject.replace('_',' '))
    for i in range(n_shots):
        prompt += shots.iloc[i, 0]
        n_choices = shots.shape[1] - 2
        for j in range(n_choices):
            prompt += "\n{}. {}".format(chr(ord('A')+j), shots.iloc[i, j+1])
        prompt += "\nAnswer: {}\n\n".format(shots.iloc[i,-1])
    
    prompt += sample.iloc[0]
    n_choices = sample.shape[0] - 2
    for j in range(n_choices):
        prompt += "\n{}. {}".format(chr(ord('A')+j), sample.iloc[j+1])
    prompt += "\nAnswer:"

    return prompt

def build_customized_prompt(subject, sample):
    prompt = ""
    prompt += "The following is a single-choice question about {}. ".format(subject.replace('_', ' '))
    prompt += "Please try to think hard and make sure you figure out the right anwser. "
    prompt += "You don't have to explain your choice. Just show your choice in a single capital letter. \n\n"
    prompt += sample.iloc[0]
    n_choices = sample.shape[0] - 2
    for j in range(n_choices):
        prompt += '\n{}. {}'.format(chr(ord('A')+j), sample.iloc[j+1])
    return prompt

def internlm_prompt_wrapper(query):
    return '<|User|>:{}\n<|Bot|>:'.format(query)

def sample_requests(args:argparse.Namespace):
    sampled_requests = list()
    subjects = sorted([f.split("_test.csv")[0] for f in os.listdir(os.path.join(args.data_dir, "test")) if "_test.csv" in f])
    for subject in subjects:
        test_df = pandas.read_csv(os.path.join(args.data_dir, "test", subject + "_test.csv"), header=None)
        train_df = pandas.read_csv(os.path.join(args.data_dir, "dev", subject + "_dev.csv"), header=None)[:args.n_shots]
        for i, sample in test_df.iterrows():
            if args.prompt_format == 'standard':
                query = build_standard_prompt(subject, sample, train_df)
            elif args.prompt_format == 'customized':
                query = build_customized_prompt(subject, sample)
            else:
                raise "Not supported prompt format!"
            query = internlm_prompt_wrapper(query)
            answer = sample.iloc[-1]
            request = dict(subject=subject, index=i, query=query, answer=answer)
            sampled_requests.append(request)
    if args.max_request_num > 0 and len(sampled_requests) > args.max_request_num:
        sampled_requests = random.sample(sampled_requests, args.max_request_num)
    print(f"Total number of requests:", len(sampled_requests))
    return sampled_requests


async def send_request_async(url, request):
    headers = {'Content-Type': 'application/json'}
    data = {
        "inputs": request['query'],
        "parameters": {
            "do_sample": False,
            "ignore_eos": False,
            "max_new_tokens": 1,
            # "repetition_penalty": 1.0, 
            # "temperature": 0.8,
            # "top_p": 0.8,
        }
    }
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, partial(requests.post, url=url, headers=headers, data=json.dumps(data)))
    if response.status_code == 200:
        response = response.json()
        print('SUCEESS', request, response)
        return request, response
    else:
        print('FAILIURE', request, response)
        return request, None

def send_requests_async(args, requests):
    tasks = list()
    for request in requests:
        tasks.append(send_request_async(args.service_url, request))
    loop = asyncio.get_event_loop()
    done, _ = loop.run_until_complete(asyncio.wait(tasks))
    results = list(map(lambda t: t.result(), done))    
    loop.close()
    return results


def process_results(args, results):
    correct = defaultdict(int)
    total = defaultdict(int)
    for request, response in results:
        if response is None:
            continue
        expected_answer = request['answer']
        actual_answer = response['generated_text'][0].strip()
        if expected_answer == actual_answer:
            correct['overall'] += 1
            correct[request['subject']] += 1
        total['overall'] += 1
        total[request['subject']] += 1
    
    summary = []
    for k in total.keys():
        summary.append({
            'subject': k,
            'correct': correct[k],
            'total': total[k],
            'accuracy': correct[k] / total[k]
        })
    summary = pandas.DataFrame(summary)
    summary.to_csv(os.path.join(args.save_dir, f'{args.description}.csv'))
    print(summary)


def main(args):
    # if not os.path.exists(args.save_dir):
    #     os.mkdir(args.save_dir)
    requests = sample_requests(args)
    results = send_requests_async(args, requests)
    process_results(args, results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_shots", type=int, default=5)
    parser.add_argument("--data_dir", type=str, default="/home/mnt/liyonghua1/datasets/mmlu")
    parser.add_argument("--save_dir", type=str, default="results")
    parser.add_argument("--service_url", type=str, default="http://10.119.1.225:8080/generate")
    parser.add_argument("--max_request_num", type=int, default=-1)
    parser.add_argument("--description", type=str, default='benchmark')
    parser.add_argument("--prompt_format", type=str, default='standard')
    args = parser.parse_args()
    main(args)