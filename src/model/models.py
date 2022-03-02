# -*- coding: utf-8 -*-

import os
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import BertPreTrainedModel, BertModel


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


seed_everything(2022)


class BERT_CRF(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BERT_CRF, self).__init__(config)

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, tags=None):

        sequence_output, pooled_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        sequence_output = self.dropout(sequence_output)
        emissions = self.classifier(sequence_output)
        loss = -1 * self.crf(emissions, tags, mask=attention_mask.byte())

        return loss

    def predict(self, input_ids=None, token_type_ids=None, attention_mask=None):

        sequence_output, pooled_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        emissions = self.classifier(sequence_output)

        return self.crf.decode(emissions, attention_mask.byte())


class BERT_BiLSTM_CRF(BertPreTrainedModel):
    def __init__(self, config, args):
        super(BERT_BiLSTM_CRF, self).__init__(config)

        self.num_layers = args.lstm_num_layers
        self.rnn_dim = args.lstm_hidden_size

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.lstm = nn.LSTM(input_size=config.hidden_size, hidden_size=self.rnn_dim, num_layers=self.num_layers,
                            bidirectional=True, batch_first=True)
        self.classifier = nn.Linear(self.rnn_dim * 2, config.num_labels)

        self.crf = CRF(num_tags=config.num_labels, batch_first=True)

    def init_hidden(self, batch_size):
        h0 = torch.randn(2 * self.num_layers, batch_size, self.rnn_dim, requires_grad=True)
        c0 = torch.randn(2 * self.num_layers, batch_size, self.rnn_dim, requires_grad=True)
        return h0, c0

    def forward(self, input_ids=None, token_type_ids=None, attention_mask=None, labels=None):

        sequence_output, pooled_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        # init lstm hidden states
        h0, c0 = self.init_hidden(input_ids.shape[0])
        init_hidden = h0.to(input_ids.device), c0.to(input_ids.device)

        sequence_output = self.dropout(sequence_output)
        sequence_output, _ = self.lstm(sequence_output, init_hidden)
        emissions = self.classifier(sequence_output)
        loss = -1 * self.crf(emissions, labels, mask=attention_mask.byte())

        return loss

    def predict(self, input_ids=None, token_type_ids=None, attention_mask=None):

        sequence_output, pooled_output = self.bert(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            return_dict=False
        )

        sequence_output, _ = self.lstm(sequence_output)
        emissions = self.classifier(sequence_output)

        result = self.crf.decode(emissions, attention_mask.byte())

        return result
