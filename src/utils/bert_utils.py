# coding:utf-8

import os
import random
import numpy as np
import torch
from collections import defaultdict
from torch.optim import Optimizer, AdamW
from torch.optim.lr_scheduler import LambdaLR
from transformers import BertTokenizer, BertConfig

from src.model.models import BERT_BiLSTM_CRF, BERT_CRF


class WarmupLinearSchedule(LambdaLR):
    def __init__(self, optimizer, warmup_steps, t_total, last_epoch=-1):
        self.warmup_steps = warmup_steps
        self.t_total = t_total
        super(WarmupLinearSchedule, self).__init__(optimizer, self.lr_lambda, last_epoch=last_epoch)

    def lr_lambda(self, step):
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        return max(0.0, float(self.t_total - step) / float(max(1.0, self.t_total - self.warmup_steps)))


class EMA:
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class Lookahead(Optimizer):
    def __init__(self, optimizer, k=5, alpha=0.5):
        self.optimizer = optimizer
        self.k = k
        self.alpha = alpha
        self.param_groups = self.optimizer.param_groups
        self.state = defaultdict(dict)
        self.fast_state = self.optimizer.state
        for group in self.param_groups:
            group["counter"] = 0

    def update(self, group):
        for fast in group["params"]:
            param_state = self.state[fast]
            if "slow_param" not in param_state:
                param_state["slow_param"] = torch.zeros_like(fast.data)
                param_state["slow_param"].copy_(fast.data)
            slow = param_state["slow_param"]
            slow += (fast.data - slow) * self.alpha
            fast.data.copy_(slow)

    def update_lookahead(self):
        for group in self.param_groups:
            self.update(group)

    def step(self, closure=None):
        loss = self.optimizer.step(closure)
        for group in self.param_groups:
            if group["counter"] == 0:
                self.update(group)
            group["counter"] += 1
            if group["counter"] >= self.k:
                group["counter"] = 0
        return loss

    def state_dict(self):
        fast_state_dict = self.optimizer.state_dict()
        slow_state = {
            (id(k) if isinstance(k, torch.Tensor) else k): v
            for k, v in self.state.items()
        }
        fast_state = fast_state_dict["state"]
        param_groups = fast_state_dict["param_groups"]
        return {
            "fast_state": fast_state,
            "slow_state": slow_state,
            "param_groups": param_groups,
        }

    def load_state_dict(self, state_dict):
        slow_state_dict = {
            "state": state_dict["slow_state"],
            "param_groups": state_dict["param_groups"],
        }
        fast_state_dict = {
            "state": state_dict["fast_state"],
            "param_groups": state_dict["param_groups"],
        }
        super(Lookahead, self).load_state_dict(slow_state_dict)
        self.optimizer.load_state_dict(fast_state_dict)
        self.fast_state = self.optimizer.state

    def add_param_group(self, param_group):
        param_group["counter"] = 0
        self.optimizer.add_param_group(param_group)


class FGM:
    def __init__(self, args, model):
        self.model = model
        self.backup = {}
        self.emb_name = args.emb_name
        self.epsilon = args.epsilon

    def attack(self):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class PGD:
    def __init__(self, args, model):
        self.model = model
        self.emb_backup = {}
        self.grad_backup = {}
        self.epsilon = args.epsilon
        self.emb_name = args.emb_name
        self.alpha = args.alpha

    def attack(self, is_first_attack=False):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_first_attack:
                    self.emb_backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = self.alpha * param.grad / norm
                    param.data.add_(r_at)
                    param.data = self.project(name, param.data, self.epsilon)

    def restore(self):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]
        self.emb_backup = {}

    def project(self, param_name, param_data, epsilon):
        r = param_data - self.emb_backup[param_name]
        if torch.norm(r) > epsilon:
            r = epsilon * r / torch.norm(r)
        return self.emb_backup[param_name] + r

    def backup_grad(self):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and param.grad is not None:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        for name, param in self.model.bert.named_parameters():
            if param.requires_grad and param.grad is not None:
                param.grad = self.grad_backup[name]


# def build_optimizer(args, model, train_steps):
#     no_decay = ['bias', 'LayerNorm.weight']
#
#     bert_param_optimizer = list(model.bert.named_parameters())
#     crf_param_optimizer = list(model.crf.named_parameters())
#     other_param_optimizer = list(model.classifier.named_parameters()) + list(model.lstm.named_parameters())
#
#     optimizer_grouped_parameters = [
#         {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
#          'weight_decay_rate': args.weight_decay},
#         {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
#          'weight_decay_rate': 0.0},
#
#         {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
#          'weight_decay_rate': args.weight_decay, 'lr': args.crf_learning_rate},
#         {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
#          'weight_decay_rate': 0.0},
#
#         {'params': [p for n, p in other_param_optimizer if not any(nd in n for nd in no_decay)],
#          'weight_decay_rate': args.weight_decay, 'lr': args.other_learning_rate},
#         {'params': [p for n, p in other_param_optimizer if any(nd in n for nd in no_decay)],
#          'weight_decay_rate': 0.0}
#     ]
#
#     optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
#     if args.use_lookahead:
#         optimizer = Lookahead(optimizer, 5, 1)
#     scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * args.warmup_ratio, t_total=train_steps)
#
#     return optimizer, scheduler


def build_optimizer(args, model, train_steps):
    no_decay = ['bias', 'LayerNorm.weight']

    bert_param_optimizer = list(model.bert.named_parameters())
    crf_param_optimizer = list(model.crf.named_parameters())

    optimizer_grouped_parameters = [
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.weight_decay},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0},

        {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay_rate': args.weight_decay, 'lr': args.crf_learning_rate},
        {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)],
         'weight_decay_rate': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    if args.use_lookahead:
        optimizer = Lookahead(optimizer, 5, 1)
    scheduler = WarmupLinearSchedule(optimizer, warmup_steps=train_steps * args.warmup_ratio, t_total=train_steps)

    return optimizer, scheduler


# def build_model_and_tokenizer(args, num_labels):
#     tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
#     bert_config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
#     model = BERT_BiLSTM_CRF.from_pretrained(args.model_name_or_path, config=bert_config, args=args)
#     model.to(args.device)
#
#     return tokenizer, model


def build_model_and_tokenizer(args, num_labels):
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case)
    bert_config = BertConfig.from_pretrained(args.model_name_or_path, num_labels=num_labels)
    model = BERT_CRF.from_pretrained(args.model_name_or_path, config=bert_config, args=args)
    model.to(args.device)

    return tokenizer, model


def batch2cuda(args, batch, test=False):
    if test:
        input_ids, origin_tokens, token_type_ids, attention_mask = \
            batch['input_ids'], batch['origin_tokens'], batch['token_type_ids'], batch['attention_mask']
        input_ids, token_type_ids, attention_mask = \
            input_ids.to(args.device), token_type_ids.to(args.device), attention_mask.to(args.device)

        return input_ids, origin_tokens, token_type_ids, attention_mask

    else:
        input_ids, origin_tokens, token_type_ids, attention_mask, labels = \
            batch['input_ids'], batch['origin_tokens'], batch['token_type_ids'], \
            batch['attention_mask'], batch['labels']
        input_ids, token_type_ids, attention_mask, labels = \
            input_ids.to(args.device), token_type_ids.to(args.device), \
            attention_mask.to(args.device), labels.to(args.device)

        return input_ids, origin_tokens, token_type_ids, attention_mask, labels


def save_model(args, model, tokenizer):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_to_save.save_pretrained(args.output_path)
    tokenizer.save_vocabulary(args.output_path)

    torch.save(args, os.path.join(args.output_path, 'training_config.bin'))


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
