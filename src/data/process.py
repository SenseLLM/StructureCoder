import os
import sys
import ast
import json
import time
import random
import argparse
import tempfile
import traceback
import subprocess

from multiprocessing import Pool

sys.setrecursionlimit(10000)

os.chdir(os.path.dirname(os.path.abspath(__file__)))

def load_jsonl(path):
    if path.endswith('.json'):
        with open(path, 'r', encoding='utf-8') as fr:
            return json.load(fr)
    data = []
    with open(path, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            data.append(json.loads(line))
    return data

def save_jsonl(data, path, mode='w'):
    with open(path, mode, encoding='utf-8') as fw:
        for d in data:
            fw.write(json.dumps(d, ensure_ascii=False) + '\n')

def format_code(data):

    if 'processed_solutions' in data:
        return data

    data['solutions'] = json.loads(data['solutions'])
    data['processed_solutions'] = []

    for solution in data['solutions']:
        try:
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
                temp_file.write(solution)
                file_name = temp_file.name

            result = subprocess.run(['black', '--quiet', '--fast', temp_file.name], capture_output=True, text=True)

            if 'error' in result.stderr + result.stdout:
                solution = "BLACK ERROR:" + result.stderr + result.stdout
            else:
                with open(temp_file.name, 'r') as fr:
                    solution = fr.read()
        except:
            traceback.print_exc()
            continue
        finally:
            os.remove(file_name)
        
        data['processed_solutions'].append(solution)

    return [data]

class BlockVisitor(ast.NodeVisitor):

    def __init__(self):
        self.blocks = []

    def generic_visit(self, node):
        depth = 0
        for _, value in ast.iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, ast.AST):
                        depth = max(self.visit(item), depth)
            elif isinstance(value, ast.AST):
                depth = max(self.visit(value), depth)
        
        return depth

    def visit_If(self, node):
        depth = self.generic_visit(node) + 1
        self.blocks.append((node.lineno, node.end_lineno, 'if', depth))
        
        return depth
        
    def visit_For(self, node):
        depth = self.generic_visit(node) + 1
        self.blocks.append((node.lineno, node.end_lineno, 'for', depth))

        return depth
    
    def visit_While(self, node):
        depth = self.generic_visit(node) + 1
        self.blocks.append((node.lineno, node.end_lineno, 'while', depth))

        return depth

    def visit_FunctionDef(self, node):
        depth = self.generic_visit(node) + 1
        self.blocks.append((node.lineno, node.end_lineno, 'def', depth))
        return depth

def fill_in_the_middle_prompt(code):
    code = code.split('\n')
    outs = [] 
    for c in code:
        if c.strip().startswith('#'):
            continue
        if '#' in c:
            c = c.split('#')[0].rstrip()
        outs.append(c)
    code = '\n'.join(outs)

    try:
        tree = ast.parse(code)
        visitor = BlockVisitor()
        visitor.visit(tree)

        if len(visitor.blocks) == 0:
            return []
  
        results = [(b[0] - 1, b[1] - 1, b[2], b[3]) for b in visitor.blocks]

        codes, outs = code.split('\n'), []
        for r in results:
            pre = '\n'.join(codes[:r[0]]) + '\n'
            mid = '\n'.join(codes[r[0]:r[1] + 1])
            suf = '\n' + '\n'.join(codes[r[1] + 1:])

            if mid.strip().startswith('elif'):
                continue

            if r[2] == 'def' and '_' in mid.split('\n')[0].split('(')[0]:
                continue

            outs.append(dict(pre=pre, mid=mid, suf=suf, type=r[2], depth=r[3]))
        
        return outs
    except:
        return []

def full(data):
    if max(data['result']) != 1:
        return []

    right_solution = None
    for solution, r in zip(data['processed_solutions'], data['result']):
        if r == 1:
            right_solution = solution
            break
    
    data.pop('solutions')
    data.pop('result')
    data.pop('processed_solutions')

    data['solution'] = right_solution

    return [data]

def fim(data):

    outs = []
    for solution, r in zip(data['processed_solutions'], data['result']):
        if r != 1 or 'BLACK ERROR' in solution or len(solution.split()) > 2048:
            continue
            
        solution = solution.strip()

        results = fill_in_the_middle_prompt(solution)
        for split in results:
            local_d = data.copy()

            for k in ['solutions', 'processed_solutions', 'result']:
                local_d.pop(k)

            local_d.update(split)
            outs.append(local_d)
    
    return outs

def random_line(data):

    outs = []
    for solution, r in zip(data['processed_solutions'], data['result']):
        if r != 1 or 'BLACK ERROR' in solution or len(solution.split()) > 2048:
            continue
            
        solution = solution.strip()

        code = [] 
        for c in solution.split('\n'):
            if c.strip().startswith('#'):
                continue
            if '#' in c:
                c = c.split('#')[0].rstrip()
            code.append(c)

        for _ in range(5):
            try:
                random_i = random.randint(0, len(code) - 10)
                random_j = random.randint(random_i + 1, random_i + 8)

                local_d = data.copy()

                for k in ['solutions', 'processed_solutions', 'result']:
                    local_d.pop(k)

                local_d['pre'] = '\n'.join(code[:random_i]) + '\n'
                local_d['mid'] = '\n'.join(code[random_i:random_j])
                local_d['suf'] = '\n' + '\n'.join(code[random_j:])

                outs.append(local_d)
            except:
                pass
    
    return outs
            
def process():
    data = load_jsonl(args.in_path)

    start, total, post_outs = time.time(), len(data), []
    with Pool(32) as pool:
        results = pool.imap(globals()[args.task], data)
        for i, result in enumerate(results, start=1):
            post_outs.extend(result)
            
            if i % 10 == 0:
                t = time.time() - start
                all_time = t * ((total) / i)
                rest_time = all_time - t 
                print(f"{i}/{total} processed. {len(post_outs)} cases. {t:.2f}/{all_time:.2f} ({rest_time:.2f}) sec", flush=True)
    
    rng = random.Random(42)
    rng.shuffle(post_outs)
    
    print(len(post_outs))
    save_jsonl(post_outs, args.out_path)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()

    parser.add_argument('-t', '--task', default='fim', type=str)

    parser.add_argument('-i', '--in_path', default='source/apps_train_format_check.jsonl', type=str)
    parser.add_argument('-o', '--out_path', default='source/apps_fim.jsonl', type=str)

    args = parser.parse_args()

    process()
