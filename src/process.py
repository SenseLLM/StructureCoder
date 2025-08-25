import os
import json
import glob
import random
import argparse
import Levenshtein

from tqdm import tqdm

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as fr:
        for line in fr.readlines():
            try:
                data.append(json.loads(line))
            except:
                pass
    return data

def save_jsonl(data, path, mode='w'):
    with open(path, mode, encoding='utf-8') as fw:
        for d in data:
            fw.write(json.dumps(d, ensure_ascii=False) + '\n')

def construct_dpo(data, out_path):
    outs = []

    rng = random.Random(42)

    for d in tqdm(data):
        right, wrong = set(), set()
        for r, c in zip(d['result_post'], d['output_post']):
            c = c.strip('\n')
            
            if r == 1:
                right.add(c)
            
            if r == 0:
                wrong.add(c)

        if len(wrong) == 0 or len(right) == 0:
            continue

        w = wrong[0]
        r, min_dis = None, float('inf')
        for _r in list(right):
            dis = Levenshtein.distance(_r, w)
            if dis < min_dis:
                r, min_dis = _r, dis
        
        depth = len(d['mid'].split('\n'))

        if len(outs) < depth:
            for _ in range(depth - len(outs)):
                outs.append([])

        outs[depth - 1].append(dict(
            chosen=dict(
                query=d['question'], 
                response_pre=d['pre'], 
                response_mid=r, 
                response_suf=d['suf']
            ),
            rejected=dict(
                query=d['question'], 
                response_pre=d['pre'], 
                response_mid=w, 
                response_suf=d['suf']
            )
        ))
    
    rng = random.Random(42)

    merged_outs = []
    for out in outs:
        if not out:
            continue

        for _ in range(args.epoch):
            rng.shuffle(out)
            merged_outs.extend(out.copy())
    
    print(len(merged_outs))

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    save_jsonl(merged_outs, out_path)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--in_path', default="data/deepseek_7B", type=str)
    parser.add_argument('-o', '--out_path', default="test.jsonl", type=str)

    parser.add_argument('-e', '--epoch', default=1, type=int)
    parser.add_argument('-t', '--task', default='fim', type=str, nargs='+')

    args = parser.parse_args()

    if isinstance(args.task, str):
        args.task = [args.task]
    
    data = []
    for task in args.task:
        for file in sorted(glob.glob(f"{args.in_path}/apps_{task}_check.jsonl.*")):
            data.extend(load_jsonl(file))

    construct_dpo(data, args.out_path)
