import os
import logging

from utils.dpo_trainer import DPOTrainer
from utils.other_utils import set_env, tokenize_dataset, get_model, get_ref_model

from dataclasses import field, dataclass
from transformers import HfArgumentParser, TrainingArguments

logger = logging.getLogger()

@dataclass
class DPOTrainingArguments(TrainingArguments):

    train_file: list[str] = field(default=None)
    
    beta: float = field(default=0.5)

    max_len: int = field(default=2048)
    num_workers: int = field(default=64)

    mode: str = field(default="qwen")
    model_cfg: str = field(default=None)

    ref_model_cfg: str = field(default=None)

    pre_loss: bool = field(default=False)
    suf_loss: bool = field(default=False)

    fim_rate: float = field(default=0.5)

    random_sampler: bool = field(default=False)

    def __post_init__(self):
        super().__post_init__()
        
        self.gradient_checkpointing_kwargs = dict(use_reentrant=False)

        if self.ref_model_cfg is None:
            self.ref_model_cfg = self.model_cfg

def train():

    parser = HfArgumentParser(DPOTrainingArguments)
    
    training_args = parser.parse_args_into_dataclasses()[0]
    
    set_env(training_args)

    model, tokenizer, processor = get_model(training_args)

    train_sets = tokenize_dataset(training_args, processor, tokenizer, training_args.train_file)

    ref_model = get_ref_model(training_args)

    trainer = DPOTrainer(
        ref_model,
        model=model, 
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_sets,
    )

    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))

if __name__ == "__main__":
    
    try:
        train()
    except Exception as e:
        logging.exception(e)
