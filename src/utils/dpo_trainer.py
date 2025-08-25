import torch
import logging
import datetime

import deepspeed
import torch.nn.functional as F
from torch.utils.data import SequentialSampler

from .other_utils import IGNORE_INDEX

from copy import deepcopy
from transformers import Trainer
from dataclasses import dataclass
from transformers import TrainerCallback
from torch.nn.utils.rnn import pad_sequence
from transformers.data.data_collator import DataCollatorMixin
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

logger = logging.getLogger()

class LoggerCallback(TrainerCallback):

    def on_train_begin(self, args, state, control, **kwargs):
        
        self.start_time = datetime.datetime.now()
        self.start_step = state.global_step

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_local_process_zero:
            return
        
        if 'loss' not in logs and 'eval_loss' not in logs:
            return
        
        loss_msg = '\n\t'.join(["%s: %.4f" % (k, v) for k, v in logs.items() if k not in ['grad_norm', 'epoch', 'learning_rate']])
        now = datetime.datetime.now()
        pass_time = now - self.start_time

        max_step = state.max_steps - self.start_step
        cur_step = state.global_step - self.start_step

        rest_time = pass_time * (max_step - cur_step) / cur_step
        eta = now + rest_time

        pt_min = pass_time.seconds // 60
        pass_time = '%.2d:%.2d' % (pt_min // 60 + pass_time.days * 24, pt_min % 60)

        rt_min = rest_time.seconds // 60
        rest_time = '%.2d:%.2d' % (rt_min // 60 + rest_time.days * 24, rt_min % 60)

        logger.info(
            'step: %d epoch: %.2f lr: %.4g passed time: %s rest time: %s eta: %s\n\t%s',
            state.global_step, state.epoch, logs.get('learning_rate', 0),
            pass_time, rest_time, eta.strftime('%m/%d %H:%M'), loss_msg
        )

@dataclass
class DPOPadCollator(DataCollatorMixin):

    tokenizer: PreTrainedTokenizerBase

    def pad(self, inputs, padding_value):

        outputs = pad_sequence(inputs, batch_first=True, padding_value=padding_value)

        return outputs.long()

    def __call__(self, inputs):

        input_ids = [torch.tensor(i[p + 'input_ids']) for p in ['positive_', 'negative_'] for i in inputs]
        labels = [torch.tensor(i[p + 'labels']) for p in ['positive_', 'negative_'] for i in inputs]

        results = {
            "input_ids": self.pad(input_ids, self.tokenizer.pad_token_id),
            "labels_not_cal_items_per_batch": self.pad(labels, IGNORE_INDEX)
        }

        return results

class DPOTrainer(Trainer):
    
    def __init__(self, ref_model, *args, **kwargs):

        super().__init__(*args, **kwargs)

        self.add_callback(LoggerCallback)

        self.data_collator = DPOPadCollator(self.tokenizer)

        self.ref_model = ref_model
        
        for param in self.ref_model.parameters():
            param.requires_grad = False

        if self.is_deepspeed_enabled:
            self.ref_model = self._prepare_deepspeed(self.ref_model)
        else:
            self.ref_model.cuda(self.model.device)

        self.beta = self.args.beta

        self._stored_metrics = {}
    
    def _prepare_deepspeed(self, model):
        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)

        if model is not None:
            if hasattr(model, "config"):
                hidden_size = (
                    max(model.config.hidden_sizes)
                    if getattr(model.config, "hidden_sizes", None)
                    else getattr(model.config, "hidden_size", None)
                )
                if hidden_size is not None and config_kwargs["zero_optimization"]["stage"] == 3:
                    config_kwargs["zero_optimization"].update(
                        {
                            "reduce_bucket_size": hidden_size * hidden_size,
                            "stage3_param_persistence_threshold": 10 * hidden_size,
                            "stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                        }
                    )

        if 'optimizer' in config_kwargs:
            config_kwargs.pop('optimizer')

        if config_kwargs["zero_optimization"]["stage"] != 3:
            config_kwargs["zero_optimization"]["stage"] = 0
        
        model, *_ = deepspeed.initialize(model=model, config=config_kwargs)

        model.eval()

        return model
    
    def _get_train_sampler(self):
        if self.args.random_sampler:
            return super()._get_train_sampler()
        return SequentialSampler(self.train_dataset)
    
    def forward(self, model, batch):

        len_postive = batch["input_ids"].shape[0] // 2

        all_logits = model(batch["input_ids"], use_cache=False).logits

        local_labels = batch["labels"].clone()

        loss_mask = (batch["labels"] != IGNORE_INDEX).float()
        local_labels[local_labels == IGNORE_INDEX] = 0
        
        per_token_logps = torch.gather(all_logits.log_softmax(-1), dim=2, index=local_labels.unsqueeze(2)).squeeze(2)

        all_logps = (per_token_logps * loss_mask).sum(-1)

        return all_logps[:len_postive], all_logps[len_postive:]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):

        inputs['labels'] = inputs['labels_not_cal_items_per_batch']
        inputs['labels'][:, 0] = IGNORE_INDEX
        inputs['labels'] = inputs['labels'].roll(shifts=-1, dims=1)

        policy_chosen_logps, policy_rejected_logps = self.forward(model, inputs)

        with torch.no_grad():
            reference_chosen_logps, reference_rejected_logps = self.forward(self.ref_model, inputs)
        
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps
        logratios = pi_logratios - ref_logratios

        loss = - F.logsigmoid(self.beta * logratios).mean()

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()
        
        reward_margins = chosen_rewards - rejected_rewards
        reward_accuracies = (chosen_rewards > rejected_rewards).float()

        self.store_metrics({
            "rewards/chosen": self._nested_gather(chosen_rewards.mean()).mean().item(),
            "rewards/rejected": self._nested_gather(rejected_rewards.mean()).mean().item(),
            "rewards/accuracies": self._nested_gather(reward_accuracies.mean()).mean().item(),
            "rewards/margins": self._nested_gather(reward_margins.mean()).mean().item(),
            "logps/chosen": self._nested_gather(policy_chosen_logps.mean()).mean().item(),
            "logps/rejected": self._nested_gather(policy_rejected_logps.mean()).mean().item(),
            "logps/ref_chosen": self._nested_gather(reference_chosen_logps.mean()).mean().item(),
            "logps/ref_rejected": self._nested_gather(reference_rejected_logps.mean()).mean().item(),
        })

        return (loss, None) if return_outputs else loss

    def store_metrics(self, metrics):
        for key, value in metrics.items():
            if key not in self._stored_metrics:
                self._stored_metrics[key] = []
            self._stored_metrics[key].append(value)
    
    def log(self, logs, start_time=None):
        for key, metrics in self._stored_metrics.items():
            if 'eval_loss' in logs:
                key = 'eval_' + key
            logs[key] = torch.tensor(metrics).mean().item()
            self._stored_metrics[key].clear()
        super().log(logs, start_time)