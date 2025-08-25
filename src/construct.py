import os
import re
import json
import torch
import argparse

import utils.template as template
from vllm import LLM, SamplingParams

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as fr:
        for line in fr.readlines():
            try:
                data.append(json.loads(line))
            except:
                print(line)
    return data

def save_jsonl(data, path, mode='w'):
    with open(path, mode, encoding='utf-8') as fw:
        for d in data:
            fw.write(json.dumps(d, ensure_ascii=False) + '\n')

def extract_type(mid):
    mid = mid.strip('\n').split('\n')

    _type = re.split(r'\s', mid[0].strip())[0]

    assert _type in ['if', 'for', 'while', 'def'], _type
    _type = mid[0][:-len(mid[0].lstrip(' '))] + _type
    
    return _type

def post(output, type):
    output = output.strip('\n').split('\n')

    indent = len(output[0]) - len(output[0].lstrip(' '))
    final_code = output[:1]
    for c in output[1:]:
        if len(c.strip()) == 0:
            final_code.append(c)
            continue

        this_indent = len(c) - len(c.lstrip(' '))
        this_c_strip = c.strip()

        if this_indent == indent and type == 'if' \
            and (this_c_strip.startswith('else') or this_c_strip.startswith('elif')):
            final_code.append(c)
            continue
        
        if this_indent <= indent:
            break

        final_code.append(c)
    
    final_code = '\n'.join(final_code)

    return final_code.rstrip()

def transfer(prompt):
    return '\n'.join([f"# {p}" if p.strip() else "" for p in prompt.split('\n')])

def consturct_fim(d, start):
    base = getattr(template, args.mode)

    fim_prompt = transfer(d['question'])
    return f"{base['pre']}{fim_prompt}\n\n{d['pre']}{base['suf']}{d['suf']}{base['mid']}{start}"

def construct_full(d, start):
    base = getattr(template, args.mode)

    return base['question'].format(question=d['question']) + f"```python{start}"

def full(in_path, out_path):

    data = load_jsonl(in_path)

    prompts = []
    for d in data:
        prompts.append(construct_full(d, start=d['starter_code'].strip()))
    print(prompts[0])
    print(len(prompts))

    completions = model.generate(prompts, generate_params)
    print(completions[0])

    outs = []
    for d, completion in zip(data, completions):
        output_src, output_post = [], []
        for output in completion.outputs:
            text = d['starter_code'].strip() + output.text

            post_output = text

            for string in ['\n```']:
                if string in post_output:
                    post_output = post_output[:post_output.find(string)]

            output_src.append(text)
            output_post.append(post_output)
        
        d['pre'] = ''
        d['suf'] = ''
        d['mid'] = d['solution']

        d['output_src'] = output_src
        d['output_post'] = output_post
        outs.append(d)
    
    save_jsonl(outs, out_path)

def fim(in_path, out_path):

    data = load_jsonl(in_path)

    outs = []
    for d in data:
        if args.depth < 0:
            outs.append(d)
        elif d['depth'] == args.depth:
            outs.append(d)
        elif args.max_depth == args.depth and d['depth'] >= args.max_depth:
            outs.append(d)
    data = outs

    prompts = []
    for d in data:
        prompts.append(consturct_fim(d, start=extract_type(d['mid'])))
    print(prompts[0])
    print(len(prompts))

    completions = model.generate(prompts, generate_params)
    print(completions[0])

    outs = []
    for d, completion in zip(data, completions):
        output_src, output_post = [], []
        for output in completion.outputs:
            text = extract_type(d['mid']) + output.text

            post_output = post(text, d['type'])
            output_src.append(text)
            output_post.append(post_output)

        d['output_src'] = output_src
        d['output_post'] = output_post

        outs.append(d)
    
    save_jsonl(outs, out_path)

def random(in_path, out_path):

    data = load_jsonl(in_path)

    prompts = []
    for d in data:
        prompts.append(consturct_fim(d, start=""))
    print(prompts[0])
    print(len(prompts))

    completions = model.generate(prompts, generate_params)
    print(completions[0])

    outs = []
    for d, completion in zip(data, completions):
        output_src, output_post = [], []
        for output in completion.outputs:
            output_src.append(output.text)
            output_post.append(output.text)

        d['output_src'] = output_src
        d['output_post'] = output_post

        outs.append(d)
    
    save_jsonl(outs, out_path)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--task', default='fim', type=str)
    parser.add_argument('-p', '--path', default='models/qwen_1B', type=str)

    parser.add_argument('-i', '--in_path', default='test.jsonl', type=str)
    parser.add_argument('-o', '--out_path', default='test_out.jsonl', type=str)

    parser.add_argument('-m', '--mode', default='qwen', type=str)

    parser.add_argument('-d', '--depth', default=-1, type=int)
    parser.add_argument('-md', '--max_depth', default=-1, type=int)

    parser.add_argument('-n', '--num_gen', default=5, type=int)

    parser.add_argument('--top_p', default=0.95, type=float)
    parser.add_argument('--temperature', default=0.7, type=float)

    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out_path)), exist_ok=True)

    stop = getattr(template, args.mode)['stop']

    generate_params = SamplingParams(
        n=args.num_gen,
        top_p=args.top_p,
        temperature=args.temperature,
        max_tokens=512,
        skip_special_tokens=False,
        spaces_between_special_tokens=False,
        stop=stop,
    )

    model = LLM(
        model=args.path, 
        tensor_parallel_size=torch.cuda.device_count(), 
        trust_remote_code=True
    )

    globals()[args.task](args.in_path, args.out_path)
