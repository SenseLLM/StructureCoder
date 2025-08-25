import os
import json
import torch
import argparse

import utils.template as template

from vllm import LLM, SamplingParams
from evalplus.data import get_human_eval_plus, get_mbpp_plus

os.environ["TOKENIZERS_PARALLELISM"] = "false"

TASK = dict()

def registry(name):

    def _registry(_class):
        TASK[name] = _class
        return _class
    
    return _registry

def load_jsonl(path):
    data = []
    with open(path, 'r', encoding='utf-8') as fr:
        for line in fr.readlines():
            data.append(json.loads(line))
    return data

def save_jsonl(data, path, mode='w'):
    with open(path, mode, encoding='utf-8') as fw:
        for d in data:
            fw.write(json.dumps(d, ensure_ascii=False) + '\n')

def post_process(code, stops=None):
    try:
        if code.count('```') % 2 == 1:
            code = code[:code.rfind('```')]
        else:
            code = code[code.find('```') + 3:]
            code = code[code.find('\n') + 1:]
    except:
        pass

    if stops is None:
        stops = ['\n# Test', '\nif', '\nassert', '\nprint', "\n```", '\ncheck']

    for string in stops:
        if string in code:
            code = code[:code.find(string)]
    
    return code

@registry('humaneval')
class Humaneval:

    name = 'humaneval'
    get_dataset_func = get_human_eval_plus

    @classmethod
    def get_prompt(cls, sample):

        prompt = f"Can you complete the following Python function?\n```python\n{sample['prompt'].strip()}\n```\n"
        starter_coder = '```python'

        return base_prompt.format(question=prompt) + starter_coder

    @classmethod
    def test(cls, model, generate_params, result_path):

        samples, task_ids, prompts = [], [], []
        for task_id, sample in cls.get_dataset_func().items():
            samples.append(sample)
            task_ids.append(task_id)
            prompts.append(cls.get_prompt(sample))
        print(prompts[0])
        
        completions = model.generate(prompts, generate_params)
        print(completions[0])

        results = []
        for task_id, completion in zip(task_ids, completions):
            for output in completion.outputs:
                results.append(dict(task_id=task_id, output=output.text, completion=post_process(output.text)))
        
        target_file = os.path.join(result_path, f'{cls.name}.jsonl')
        save_jsonl(results, target_file)

@registry('mbpp')
class MBPP(Humaneval):

    name = 'mbpp'
    get_dataset_func = get_mbpp_plus

@registry('apps')
class APPS:

    name = 'apps'

    @classmethod
    def get_dataset_func(cls):
        data = load_jsonl('data/test/apps_test.jsonl')
        return {d['problem_id']: d for d in data}
    
    @classmethod
    def get_prompt(cls, sample):

        prompt = f"You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n{sample['question']}\n\n"

        if sample['starter_code'].rstrip():
            prompt += f"You will use the following starter code to write the solution to the problem and enclose your code within delimiters.```python\n{sample['starter_code'].rstrip()}\n```\n"
        else:
            prompt += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."

        return base_prompt.format(question=prompt)
    
    @classmethod
    def test(cls, model, generate_params, result_path):

        samples, prompts = [], []
        for task_id, sample in cls.get_dataset_func().items():
            samples.append(sample)
            prompts.append(cls.get_prompt(sample))
        print(prompts[0])
        
        completions = model.generate(prompts, generate_params)
        print(completions[0])

        results = []
        for sample, prompt, completion in zip(samples, prompts, completions):
            sample['solutions'] = [
                post_process(output.text, ["\n```"]) 
                for output in completion.outputs
            ]
            sample['prompt'] = prompt
            sample['solutions_src'] = [output.text for output in completion.outputs]
            results.append(sample)
        
        target_file = os.path.join(result_path, f'{cls.name}.jsonl')
        save_jsonl(results, target_file)

@registry('livecode')
class LiveCodeBench:

    name = 'live_code_bench'

    @classmethod
    def get_dataset_func(cls):
        data = load_jsonl('data/test/livecodebench.jsonl')
        return {d['question_id']: d for d in data}
    
    @classmethod
    def get_prompt(cls, sample):

        prompt = f"You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n{sample['question_content']}\n\n"

        if sample['starter_code'].rstrip():
            prompt += f"You will use the following starter code to write the solution to the problem and enclose your code within delimiters.```python\n{sample['starter_code'].rstrip()}\n```\n"
        else:
            prompt += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."

        return base_prompt.format(question=prompt)
    
    @classmethod
    def test(cls, model, generate_params, result_path):

        samples, prompts = [], []
        for task_id, sample in cls.get_dataset_func().items():
            samples.append(sample)
            prompts.append(cls.get_prompt(sample))
        print(prompts[0])
        
        completions = model.generate(prompts, generate_params)
        print(completions[0])

        results = []
        for sample, completion in zip(samples, completions):
            results.append(dict(
                question_id=sample['question_id'], 
                code_list=[
                    post_process(output.text, ["\n```"]) 
                    for output in completion.outputs
                ]
            ))
        
        target_file = os.path.join(result_path, f'{cls.name}.json')
        with open(target_file, 'w', encoding='utf8') as fw:
            json.dump(results, fw, ensure_ascii=False, indent=4)

@registry('bigcode')
class BigCodeBench:

    name = 'big_code_bench'

    @classmethod
    def get_dataset_func(cls):
        data = load_jsonl('data/test/bigcodebench.jsonl')
        return {d['task_id']: d for d in data}
    
    @classmethod
    def get_prompt(cls, sample):

        prompt = "Please provide a self-contained Python script that solves the following problem in a markdown code block:\n" + sample['instruct_prompt'].strip()
        starter_coder = "Below is a Python script with a self-contained function that solves the problem and passes corresponding tests:\n```python"

        return base_prompt.format(question=prompt) + starter_coder
    
    @classmethod
    def test(cls, model, generate_params, result_path):

        samples, prompts = [], []
        for task_id, sample in cls.get_dataset_func().items():
            samples.append(sample)
            prompts.append(cls.get_prompt(sample))
        print(prompts[0])
        
        completions = model.generate(prompts, generate_params)
        print(completions[0])

        results = []
        for sample, completion in zip(samples, completions):
            for output in completion.outputs:
                result = sample.copy()
                result['source_gen'] = output.text
                result['solution'] = post_process(
                    output.text, 
                    ["\nif __name__", "\ndef main(", "\nprint(", "\n```", "\nassert", "\n# Test", "\n# Example"]
                )
                results.append(result)
        
        target_file = os.path.join(result_path, f'{cls.name}.jsonl')
        save_jsonl(results, target_file)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--path', required=True, type=str)
    parser.add_argument('-m', '--mode', default='qwen', type=str)

    parser.add_argument('-t', '--task', default=None, type=str, nargs='+')

    args = parser.parse_args()

    base = getattr(template, args.mode)

    base_prompt = base['question']
  
    model = LLM(
        model=args.path, 
        tensor_parallel_size=torch.cuda.device_count(), 
        trust_remote_code=True
    )

    sample_params = SamplingParams(
        stop=base['stop'],
        temperature=0,
        max_tokens=1024,
        spaces_between_special_tokens=False
    )

    if args.task is None:
        args.task = ["humaneval", "mbpp", "livecode", "bigcode", "apps"]

    if isinstance(args.task, str):
        args.task = [args.task]
    
    os.makedirs(os.path.join(args.path, 'code_results'), exist_ok=True)

    for task in args.task:
        TASK[task].test(model, sample_params, os.path.join(args.path, 'code_results'))