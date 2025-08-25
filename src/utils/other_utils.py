import os
import sys
import torch
import shutil
import random
import logging
import transformers

import utils.template as template
import torch.distributed as dist

from datasets import load_dataset, concatenate_datasets
from transformers import set_seed, AutoTokenizer, AutoModelForCausalLM

logger = logging.getLogger()

IGNORE_INDEX = -100

def transfer(prompt):
    return '\n'.join([f"# {p}" if p.strip() else "" for p in prompt.split('\n')])

class Processor:
    
    def __init__(self, tokenizer, mode, max_len, pre_loss, suf_loss, fim_rate=0):

        self.tokenizer = tokenizer
        self.max_len = max_len

        self.pre_loss = pre_loss
        self.suf_loss = suf_loss
        
        self.mode = mode

        self.fim_rate = fim_rate
        self.rng = random.Random(3407)

        assert hasattr(template, mode), f"Template {mode} not found"

        base = getattr(template, mode)

        self.question = base['question']
        self.pre = base['pre']
        self.suf = base['suf']
        self.mid = base['mid']
        self.eot = base['eot']

    def _process_tokenize(self, e, prefix='', fim=False):

        e['query'] += "\n\nRead the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs)."

        if fim:
            if self.suf_loss:
                if self.pre_loss:
                    _input = self.pre + transfer(e['query']) + '\n\n'
                    _response = e['response_pre'] + self.suf + e['response_suf'] + self.mid + e['response_mid'] + self.eot
                else:
                    _input = self.pre + transfer(e['query']) + '\n\n' + e['response_pre'] + self.suf
                    _response = e['response_suf'] + self.mid + e['response_mid'] + self.eot
            else:
                _input = self.pre + transfer(e['query']) + '\n\n' + e['response_pre'] + self.suf + e['response_suf'] + self.mid
                _response = e['response_mid'] + self.eot
        else:
            if self.suf_loss:
                if self.pre_loss:
                    _input = self.question.format(question=e['query']) + "```python\n"
                    _response = e['response_pre'] + e['response_mid'] + e['response_suf']
                else:
                    _input = self.question.format(question=e['query']) + "```python\n" + e['response_pre']
                    _response = e['response_mid'] + e['response_suf']
            else:
                if self.pre_loss:
                    _input = self.question.format(question=e['query']) + "```python\n"
                    _response = e['response_pre'] + e['response_mid']
                else:
                    _input = self.question.format(question=e['query']) + "```python\n" + e['response_pre']
                    _response = e['response_mid']

        _input_id = self.tokenizer.encode(_input)
        _response_id = self.tokenizer.encode(_response, add_special_tokens=False)

        input_ids = _input_id + _response_id
        labels = [IGNORE_INDEX] * len(_input_id) + _response_id

        return {f"{prefix}input_ids": input_ids, f"{prefix}labels": labels}

    def process_tokenize(self, e):

        fim = self.rng.random() < self.fim_rate
        
        if len(e['chosen']['response_pre']) == 0 and len(e['chosen']['response_suf']) == 0:
            fim = False

        r = self._process_tokenize(e['chosen'], 'positive_', fim)
        r.update(self._process_tokenize(e['rejected'], 'negative_', fim))

        return r

    def filter_data(self, e):
        return len(e['positive_input_ids']) <= self.max_len and len(e['negative_input_ids']) <= self.max_len

def tokenize_dataset(training_args, processor, tokenizer, files):

    datasets = []

    for file in files:
        with training_args.main_process_first(desc="tokenization"):

            dataset = load_dataset('json', data_files=file, split='train')

            logger.info('Total %d case in %s before filter', len(dataset), file)

            dataset = dataset.map(
                processor.process_tokenize,
                num_proc=training_args.num_workers,
                desc="Running tokenizer on dataset",
            )

            dataset = dataset.filter(
                processor.filter_data, 
                num_proc=training_args.num_workers
            )

            logger.info('Total %d case in %s after filter', len(dataset), file)

            datasets.append(dataset)

    datasets = concatenate_datasets(datasets)
    logger.info('Total %d case after concatenation', len(datasets))
    
    for index in [0, 100, 1000]:
        for prefix in ['positive_', 'negative_', '']:
            if f"{prefix}input_ids" not in datasets[index]:
                continue

            input_ids = datasets[index][f"{prefix}input_ids"]
            input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
            input_tokens = [
                f'{input_token}|calc'  if label >= 0 else f'{input_token}|ignore' 
                for (input_token, label) in zip(input_tokens, datasets[index][f"{prefix}labels"])
            ]

            sl = '-' * 50
            logger.info(
                "\nSample %d of the raw training set ~ '%s'\n%sInput Text%s\n%s\n%sInput Tokens%s\n%s",
                index, prefix, sl, sl, tokenizer.decode(input_ids), sl, sl, input_tokens
            )

    return datasets

def get_model(training_args):

    model_args = dict(
        pretrained_model_name_or_path=training_args.model_cfg,
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="flash_attention_2"
    )
    
    model = AutoModelForCausalLM.from_pretrained(**model_args)

    logger.info(model)

    tokenizer = AutoTokenizer.from_pretrained(training_args.model_cfg, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    processor = Processor(
        tokenizer, training_args.mode, training_args.max_len, training_args.pre_loss, 
        training_args.suf_loss, training_args.fim_rate
    )

    return model, tokenizer, processor

def get_ref_model(training_args):

    model_args = dict(
        pretrained_model_name_or_path=training_args.ref_model_cfg,
        trust_remote_code=True, 
        torch_dtype=torch.bfloat16, 
    )

    if training_args.mode != 'codestral':
        model_args['attn_implementation'] = "flash_attention_2"
        
    model = AutoModelForCausalLM.from_pretrained(**model_args)

    logger.info(model)

    return model

def barrier(training_args):
    if training_args.world_size > 1:
        dist.barrier()

def print_args(args):
    max_len = max([len(k) for k in vars(args).keys()]) + 4
    logger.info("******************* Training Arguments *******************")
    for key, val in sorted(vars(args).items()):
        keystr = "{}".format(key) + (" " * (max_len - len(key)))
        logger.info("%s -->   %s", keystr, val)
    logger.info("******************* Training Arguments *******************")
 
def set_logger(_logger, local_rank, log_file=None):
    _logger.handlers.clear()
    
    if local_rank in [-1, 0]:
        _logger.setLevel(logging.INFO)
    else:
        _logger.setLevel(logging.WARN)

    log_format = '[%(asctime)s] [Rank {} - %(levelname)s] [%(filename)s - %(lineno)d] %(message)s'.format(local_rank)
    log_format = logging.Formatter(log_format, '%Y-%m-%d %H:%M:%S')
    
    console = logging.StreamHandler()
    console.setFormatter(log_format)
    _logger.addHandler(console)
    
    if log_file is not None:

        file = logging.FileHandler(log_file, mode='a')
        file.setFormatter(log_format)
        _logger.addHandler(file)

def set_env(training_args):
    training_args._frozen = False
    
    barrier(training_args)
    
    if os.path.exists(training_args.output_dir):
        if training_args.overwrite_output_dir:
            if training_args.process_index == 0:
                shutil.rmtree(training_args.output_dir)
        else:
            index = 1
            output_dir = training_args.output_dir + f'-{index}'
            while os.path.exists(output_dir):
                index = index + 1
                output_dir = training_args.output_dir + f'-{index}'
            training_args.output_dir = output_dir
            
    training_args.logging_dir = training_args.output_dir

    barrier(training_args)
    
    os.makedirs(training_args.output_dir, exist_ok=True)
    
    barrier(training_args)
    
    node_rank = int(os.getenv('GROUP_RANK', '0'))
    log_path = os.path.join(training_args.output_dir, f'train-{node_rank}.log')
    for _logger in [logger, transformers.utils.logging.get_logger(), logging.getLogger('DeepSpeed')]:
        set_logger(_logger, training_args.local_rank, log_path)
    
    barrier(training_args)

    logger.warning("Device: %s, rank: %s, world size: %s", training_args.device, training_args.process_index, training_args.world_size)

    barrier(training_args)
    
    set_seed(training_args.seed)

    logger.info("******************* Command *******************")
    logger.info(" ".join(sys.argv))
    logger.info("******************* Command *******************")
    
    print_args(training_args)
