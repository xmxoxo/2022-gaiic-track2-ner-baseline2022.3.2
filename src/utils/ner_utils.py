# coding:utf-8


import os
import re
import sys
import torch
import pickle
import numpy as np
from torch.utils.data import DataLoader, Dataset
from collections import defaultdict, namedtuple


def build_bert_inputs(inputs, label, sentence, tokenizer, label2id):

    token_list = sentence
    label_list = label

    assert len(token_list) == len(label_list)

    tokens, origin_tokens, labels = [], [], []
    for i, word in enumerate(token_list):

        if word == ' ' or word == '':
            word = '-'

        token = tokenizer.tokenize(word)

        if len(token) > 1:
            token = [tokenizer.unk_token]

        tokens.extend(token)

        origin_tokens.append(word)
        labels.append(label_list[i])

    assert len(tokens) == len(labels)

    inputs_dict = tokenizer.encode_plus(tokens, add_special_tokens=True,
                                        return_token_type_ids=True, return_attention_mask=True)

    input_ids = inputs_dict['input_ids']
    token_type_ids = inputs_dict['token_type_ids']
    attention_mask = inputs_dict['attention_mask']

    origin_tokens = ["[CLS]"] + origin_tokens + ["[SEP]"]

    label_ids = []
    label_ids.extend([label2id["O"]])
    label_ids.extend([label2id[label] for i, label in enumerate(labels)])
    label_ids.extend([label2id["O"]])
    label_ids.extend([label2id["O"]] * (len(input_ids) - len(label_ids)))

    assert len(input_ids) == len(label_ids)

    inputs['input_ids'].append(input_ids)
    inputs['origin_tokens'].append(origin_tokens)
    inputs['token_type_ids'].append(token_type_ids)
    inputs['attention_mask'].append(attention_mask)
    inputs['labels'].append(label_ids)


def build_bert_inputs_test(inputs, sentence, tokenizer, label2id):

    token_list = sentence

    tokens, origin_tokens = [], []
    for i, word in enumerate(token_list):

        if word == ' ' or word == '':
            word = '-'

        token = tokenizer.tokenize(word)

        if len(token) > 1:
            token = [tokenizer.unk_token]

        tokens.extend(token)

        origin_tokens.append(word)

    inputs_dict = tokenizer.encode_plus(tokens, add_special_tokens=True,
                                        return_token_type_ids=True, return_attention_mask=True)

    input_ids = inputs_dict['input_ids']
    token_type_ids = inputs_dict['token_type_ids']
    attention_mask = inputs_dict['attention_mask']

    origin_tokens = ["[CLS]"] + origin_tokens + ["[SEP]"]

    inputs['input_ids'].append(input_ids)
    inputs['origin_tokens'].append(origin_tokens)
    inputs['token_type_ids'].append(token_type_ids)
    inputs['attention_mask'].append(attention_mask)


class NerDataset(Dataset):
    def __init__(self, data_dict):
        super(NerDataset, self).__init__()
        self.data_dict = data_dict

    def __getitem__(self, index: int) -> tuple:
        data = (
            self.data_dict['input_ids'][index],
            self.data_dict['origin_tokens'][index],
            self.data_dict['token_type_ids'][index],
            self.data_dict['attention_mask'][index],
            self.data_dict['labels'][index]
        )

        return data

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])


class NerDataset_test(Dataset):
    def __init__(self, data_dict):
        super(NerDataset_test, self).__init__()
        self.data_dict = data_dict

    def __getitem__(self, index: int) -> tuple:
        data = (
            self.data_dict['input_ids'][index],
            self.data_dict['origin_tokens'][index],
            self.data_dict['token_type_ids'][index],
            self.data_dict['attention_mask'][index]
        )

        return data

    def __len__(self) -> int:
        return len(self.data_dict['input_ids'])


class Collator:
    def __init__(self, max_seq_len, tokenizer):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def pad_and_truncate(self, input_ids_list, origin_tokens_list, token_type_ids_list, attention_mask_list,
                         labels_list, max_seq_len):

        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)
        labels = torch.zeros_like(input_ids)
        origin_tokens = []

        for i in range(len(input_ids_list)):

            seq_len = len(input_ids_list[i])

            tmp_origin_tokens = [self.tokenizer.pad_token_id] * max_seq_len

            if seq_len <= max_seq_len:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i], dtype=torch.long)
                token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i], dtype=torch.long)
                attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i], dtype=torch.long)
                labels[i, :seq_len] = torch.tensor(labels_list[i], dtype=torch.long)

                for j, word in enumerate(origin_tokens_list[i]):
                    if j <= max_seq_len:
                        tmp_origin_tokens[j] = word
                origin_tokens.append(tmp_origin_tokens)

            else:
                input_ids[i] = torch.tensor(input_ids_list[i][:max_seq_len - 1] + [self.tokenizer.sep_token_id],
                                            dtype=torch.long)
                token_type_ids[i] = torch.tensor(token_type_ids_list[i][:max_seq_len], dtype=torch.long)
                attention_mask[i] = torch.tensor(attention_mask_list[i][:max_seq_len], dtype=torch.long)
                labels[i] = torch.tensor(labels_list[i][:max_seq_len], dtype=torch.long)

                for j in range(max_seq_len):
                    tmp_origin_tokens[j] = origin_tokens_list[i][j]
                tmp_origin_tokens[-1] = self.tokenizer.sep_token_id
                origin_tokens.append(tmp_origin_tokens)

        return input_ids, origin_tokens, token_type_ids, attention_mask, labels

    def __call__(self, examples: list) -> dict:
        input_ids_list, origin_tokens_list, token_type_ids_list, attention_mask_list, \
        labels_list = list(zip(*examples))

        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)

        input_ids, origin_tokens, token_type_ids, attention_mask, labels = \
            self.pad_and_truncate(input_ids_list, origin_tokens_list, token_type_ids_list, attention_mask_list,
                                  labels_list, max_seq_len)

        data_dict = {
            'input_ids': input_ids,
            'origin_tokens': origin_tokens,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }

        return data_dict


class Collator_test:
    def __init__(self, max_seq_len, tokenizer):
        self.max_seq_len = max_seq_len
        self.tokenizer = tokenizer

    def pad_and_truncate(self, input_ids_list, origin_tokens_list, token_type_ids_list, attention_mask_list,
                         max_seq_len):

        input_ids = torch.zeros((len(input_ids_list), max_seq_len), dtype=torch.long)
        token_type_ids = torch.zeros_like(input_ids)
        attention_mask = torch.zeros_like(input_ids)
        origin_tokens = []

        for i in range(len(input_ids_list)):

            seq_len = len(input_ids_list[i])

            tmp_origin_tokens = [self.tokenizer.pad_token_id] * max_seq_len

            if seq_len <= max_seq_len:
                input_ids[i, :seq_len] = torch.tensor(input_ids_list[i], dtype=torch.long)
                token_type_ids[i, :seq_len] = torch.tensor(token_type_ids_list[i], dtype=torch.long)
                attention_mask[i, :seq_len] = torch.tensor(attention_mask_list[i], dtype=torch.long)

                for j, word in enumerate(origin_tokens_list[i]):
                    if j <= max_seq_len:
                        tmp_origin_tokens[j] = word
                origin_tokens.append(tmp_origin_tokens)

            else:
                input_ids[i] = torch.tensor(input_ids_list[i][:max_seq_len - 1] + [self.tokenizer.sep_token_id],
                                            dtype=torch.long)
                token_type_ids[i] = torch.tensor(token_type_ids_list[i][:max_seq_len], dtype=torch.long)
                attention_mask[i] = torch.tensor(attention_mask_list[i][:max_seq_len], dtype=torch.long)

                for j in range(max_seq_len):
                    tmp_origin_tokens[j] = origin_tokens_list[i][j]
                tmp_origin_tokens[-1] = self.tokenizer.sep_token_id
                origin_tokens.append(tmp_origin_tokens)

        origin_tokens = origin_tokens_list

        return input_ids, origin_tokens, token_type_ids, attention_mask

    def __call__(self, examples: list) -> dict:
        input_ids_list, origin_tokens_list, token_type_ids_list, attention_mask_list = list(zip(*examples))

        cur_max_seq_len = max(len(input_id) for input_id in input_ids_list)
        max_seq_len = min(cur_max_seq_len, self.max_seq_len)

        input_ids, origin_tokens, token_type_ids, attention_mask = \
            self.pad_and_truncate(input_ids_list, origin_tokens_list, token_type_ids_list, attention_mask_list,
                                  max_seq_len)

        data_dict = {
            'input_ids': input_ids,
            'origin_tokens': origin_tokens,
            'token_type_ids': token_type_ids,
            'attention_mask': attention_mask
        }

        return data_dict


def load_data(args, tokenizer):
    train_cache_pkl_path = os.path.join(args.data_cache_path, 'train.pkl')
    dev_cache_pkl_path = os.path.join(args.data_cache_path, 'dev.pkl')

    with open(train_cache_pkl_path, 'rb') as f:
        train_data = pickle.load(f)

    with open(dev_cache_pkl_path, 'rb') as f:
        dev_data = pickle.load(f)

    collate_fn = Collator(args.max_seq_len, tokenizer)

    train_dataset = NerDataset(train_data)
    dev_dataset = NerDataset(dev_data)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    dev_dataloader = DataLoader(dataset=dev_dataset, batch_size=args.batch_size, shuffle=False,
                                num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    return train_dataloader, dev_dataloader

# ====================== evaluation ============================


ANY_SPACE = '<SPACE>'


class FormatError(Exception):
    pass


class EvalCounts(object):
    def __init__(self):
        self.correct_chunk = 0  # number of correctly identified chunks
        self.correct_tags = 0  # number of correct chunk tags
        self.found_correct = 0  # number of chunks in corpus
        self.found_guessed = 0  # number of identified chunks
        self.token_counter = 0  # token counter (ignores sentence breaks)

        # counts by type
        self.t_correct_chunk = defaultdict(int)
        self.t_found_correct = defaultdict(int)
        self.t_found_guessed = defaultdict(int)


def parse_args(argv):
    import argparse
    parser = argparse.ArgumentParser()
    arg = parser.add_argument
    arg('-b', '--boundary', metavar='STR', default='-X-',
        help='sentence boundary')
    arg('-d', '--delimiter', metavar='CHAR', default=ANY_SPACE,
        help='character delimiting items in input')
    arg('-o', '--otag', metavar='CHAR', default='O',
        help='alternative outside tag')
    arg('file', nargs='?', default=None)
    return parser.parse_args(argv)


def parse_tag(t):
    m = re.match(r'^([^-]*)-(.*)$', t)
    return m.groups() if m else (t, '')


def evaluation(iterable, options=None):
    if options is None:
        options = parse_args([])  # use defaults

    counts = EvalCounts()
    num_features = None  # number of features per line
    in_correct = False  # currently processed chunks is correct until now
    last_correct = 'O'  # previous chunk tag in corpus
    last_correct_type = ''  # type of previously identified chunk tag
    last_guessed = 'O'  # previously identified chunk tag
    last_guessed_type = ''  # type of previous chunk tag in corpus

    for line in iterable:
        line = line.rstrip('\r\n')

        if options.delimiter == ANY_SPACE:
            features = line.split()
        else:
            features = line.split(options.delimiter)

        if num_features is None:
            num_features = len(features)
        elif num_features != len(features) and len(features) != 0:
            raise FormatError('unexpected number of features: %d (%d)' %
                              (len(features), num_features))

        if len(features) == 0 or features[0] == options.boundary:
            features = [options.boundary, 'O', 'O']
        if len(features) < 3:
            raise FormatError('unexpected number of features in line %s' % line)

        guessed, guessed_type = parse_tag(features.pop())
        correct, correct_type = parse_tag(features.pop())
        first_item = features.pop(0)

        if first_item == options.boundary:
            guessed = 'O'

        end_correct = end_of_chunk(last_correct, correct,
                                   last_correct_type, correct_type)
        end_guessed = end_of_chunk(last_guessed, guessed,
                                   last_guessed_type, guessed_type)
        start_correct = start_of_chunk(last_correct, correct,
                                       last_correct_type, correct_type)
        start_guessed = start_of_chunk(last_guessed, guessed,
                                       last_guessed_type, guessed_type)

        if in_correct:
            if (end_correct and end_guessed and
                    last_guessed_type == last_correct_type):
                in_correct = False
                counts.correct_chunk += 1
                counts.t_correct_chunk[last_correct_type] += 1
            elif end_correct != end_guessed or guessed_type != correct_type:
                in_correct = False

        if start_correct and start_guessed and guessed_type == correct_type:
            in_correct = True

        if start_correct:
            counts.found_correct += 1
            counts.t_found_correct[correct_type] += 1
        if start_guessed:
            counts.found_guessed += 1
            counts.t_found_guessed[guessed_type] += 1
        if first_item != options.boundary:
            if correct == guessed and guessed_type == correct_type:
                counts.correct_tags += 1
            counts.token_counter += 1

        last_guessed = guessed
        last_correct = correct
        last_guessed_type = guessed_type
        last_correct_type = correct_type

    if in_correct:
        counts.correct_chunk += 1
        counts.t_correct_chunk[last_correct_type] += 1

    return counts


def end_of_chunk(prev_tag, tag, prev_type, type_):
    chunk_end = False

    if prev_tag == 'E': chunk_end = True
    if prev_tag == 'S': chunk_end = True

    if prev_tag == 'B' and tag == 'B': chunk_end = True
    if prev_tag == 'B' and tag == 'S': chunk_end = True
    if prev_tag == 'B' and tag == 'O': chunk_end = True
    if prev_tag == 'I' and tag == 'B': chunk_end = True
    if prev_tag == 'I' and tag == 'S': chunk_end = True
    if prev_tag == 'I' and tag == 'O': chunk_end = True

    if prev_tag != 'O' and prev_tag != '.' and prev_type != type_:
        chunk_end = True

    if prev_tag == ']': chunk_end = True
    if prev_tag == '[': chunk_end = True

    return chunk_end


def start_of_chunk(prev_tag, tag, prev_type, type_):
    chunk_start = False

    if tag == 'B': chunk_start = True
    if tag == 'S': chunk_start = True

    if prev_tag == 'E' and tag == 'E': chunk_start = True
    if prev_tag == 'E' and tag == 'I': chunk_start = True
    if prev_tag == 'S' and tag == 'E': chunk_start = True
    if prev_tag == 'S' and tag == 'I': chunk_start = True
    if prev_tag == 'O' and tag == 'E': chunk_start = True
    if prev_tag == 'O' and tag == 'I': chunk_start = True

    if tag != 'O' and tag != '.' and prev_type != type_:
        chunk_start = True

    # these chunks are assumed to have length 1
    if tag == '[': chunk_start = True
    if tag == ']': chunk_start = True

    return chunk_start


def uniq(iterable):
    seen = set()
    return [i for i in iterable if not (i in seen or seen.add(i))]


Metrics = namedtuple('Metrics', 'tp fp fn prec rec fscore')


def calculate_metrics(correct, guessed, total):
    tp, fp, fn = correct, guessed - correct, total - correct
    p = 0 if tp + fp == 0 else 1. * tp / (tp + fp)
    r = 0 if tp + fn == 0 else 1. * tp / (tp + fn)
    f = 0 if p + r == 0 else 2 * p * r / (p + r)
    return Metrics(tp, fp, fn, p, r, f)


def metrics(counts):
    c = counts
    overall = calculate_metrics(
        c.correct_chunk, c.found_guessed, c.found_correct
    )
    by_type = {}
    for t in uniq(list(c.t_found_correct) + list(c.t_found_guessed)):
        by_type[t] = calculate_metrics(
            c.t_correct_chunk[t], c.t_found_guessed[t], c.t_found_correct[t]
        )
    return overall, by_type


def report(counts, out=None):
    if out is None:
        out = sys.stdout

    overall, by_type = metrics(counts)

    c = counts
    out.write('processed %d tokens with %d phrases; ' %
              (c.token_counter, c.found_correct))
    out.write('found: %d phrases; correct: %d.\n' %
              (c.found_guessed, c.correct_chunk))

    if c.token_counter > 0:
        out.write('accuracy: %6.2f%%; ' %
                  (100. * c.correct_tags / c.token_counter))
        out.write('precision: %6.2f%%; ' % (100. * overall.prec))
        out.write('recall: %6.2f%%; ' % (100. * overall.rec))
        out.write('FB1: %6.2f\n' % (100. * overall.fscore))

    for i, m in sorted(by_type.items()):
        out.write('%17s: ' % i)
        out.write('precision: %6.2f%%; ' % (100. * m.prec))
        out.write('recall: %6.2f%%; ' % (100. * m.rec))
        out.write('FB1: %6.2f  %d\n' % (100. * m.fscore, c.t_found_guessed[i]))
