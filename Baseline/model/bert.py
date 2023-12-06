import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from vocab import Vocab
import utils
from transformers import BertModel


class BertCRF(nn.Module):
    def __init__(self,  tag_vocab, dropout_rate=0.5, embed_size=256, hidden_size=256):
        super(BertCRF, self).__init__()
        self.bert = BertModel.from_pretrained('pretrained_bert_models/minibert')
        self.dropout = nn.Dropout(dropout_rate)
        self.hidden2tag = nn.Linear(self.bert.config.hidden_size, len(tag_vocab))
        self.tag_vocab = tag_vocab
        self.transition = nn.Parameter(torch.randn(len(tag_vocab), len(tag_vocab)))
        self.dropout_rate=dropout_rate
    def forward(self, sentences, tags, sen_lengths):
        mask = (sentences != 0).to(sentences.device)
        sentences = sentences.transpose(0, 1)
        sentences = self.bert(sentences)[0]
        emit_score = self.encode(sentences, sen_lengths)
        loss = self.cal_loss(tags, mask, emit_score)
        return loss

    def encode(self, sentences, sent_lengths):
        sentences = self.dropout(sentences)
        emit_score = self.hidden2tag(sentences)
        return emit_score.transpose(0, 1)

    def cal_loss(self, tags, mask, emit_score):
        """ Calculate CRF loss
        Args:
            tags (tensor): a batch of tags, shape (b, len)
            mask (tensor): mask for the tags, shape (b, len), values in PAD position is 0
            emit_score (tensor): emit matrix, shape (b, len, K)
        Returns:
            loss (tensor): loss of the batch, shape (b,)
        """
        batch_size, sent_len = tags.shape
        # calculate score for the tags
        score = torch.gather(emit_score, dim=2, index=tags.unsqueeze(dim=2)).squeeze(dim=2)  # shape: (b, len)
        score[:, 1:] += self.transition[tags[:, :-1], tags[:, 1:]]
        total_score = (score * mask.type(torch.float)).sum(dim=1)  # shape: (b,)
        # calculate the scaling factor
        d = torch.unsqueeze(emit_score[:, 0], dim=1)  # shape: (b, 1, K)
        for i in range(1, sent_len):
            n_unfinished = mask[:, i].sum()
            d_uf = d[: n_unfinished]  # shape: (uf, 1, K)
            emit_and_transition = emit_score[: n_unfinished, i].unsqueeze(dim=1) + self.transition  # shape: (uf, K, K)
            log_sum = d_uf.transpose(1, 2) + emit_and_transition  # shape: (uf, K, K)

            # print(log_sum.shape)
            # print(log_sum)
            max_v = log_sum.max(dim=1)[0].unsqueeze(dim=1)  # shape: (uf, 1, K)
            log_sum = log_sum - max_v  # shape: (uf, K, K)
            d_uf = max_v + torch.logsumexp(log_sum, dim=1).unsqueeze(dim=1)  # shape: (uf, 1, K)
            d = torch.cat((d_uf, d[n_unfinished:]), dim=0)
        d = d.squeeze(dim=1)  # shape: (b, K)
        max_d = d.max(dim=-1)[0]  # shape: (b,)
        d = max_d + torch.logsumexp(d - max_d.unsqueeze(dim=1), dim=1)  # shape: (b,)
        llk = total_score - d  # shape: (b,)
        loss = -llk  # shape: (b,)
        return loss

    def predict(self, sentences, sen_lengths):
        """
        Args:
            sentences (tensor): sentences, shape (b, len). Lengths are in decreasing order, len is the length
                                of the longest sentence
            sen_lengths (list): sentence lengths
        Returns:
            tags (list[list[str]]): predicted tags for the batch
        """
        batch_size = sentences.shape[0]
        # mask = (sentences != self.sent_vocab[self.sent_vocab.PAD])  # shape: (b, len)
        mask = (sentences != 0).to(sentences.device)  # shape: (b, len)
        sentences = sentences.transpose(0, 1)  # shape: (len, b)
        # sentences = self.embedding(sentences)  # shape: (len, b, e)
        sentences = self.bert(sentences)[0]
        emit_score = self.encode(sentences, sen_lengths)  # shape: (b, len, K)
        tags = [[[i] for i in range(len(self.tag_vocab))]] * batch_size  # list, shape: (b, K, 1)
        d = torch.unsqueeze(emit_score[:, 0], dim=1)  # shape: (b, 1, K)
        for i in range(1, sen_lengths[0]):
            n_unfinished = mask[:, i].sum()
            d_uf = d[: n_unfinished]  # shape: (uf, 1, K)
            emit_and_transition = self.transition + emit_score[: n_unfinished, i].unsqueeze(dim=1)  # shape: (uf, K, K)
            new_d_uf = d_uf.transpose(1, 2) + emit_and_transition  # shape: (uf, K, K)
            d_uf, max_idx = torch.max(new_d_uf, dim=1)
            max_idx = max_idx.tolist()  # list, shape: (nf, K)
            tags[: n_unfinished] = [[tags[b][k] + [j] for j, k in enumerate(max_idx[b])] for b in range(n_unfinished)]
            d = torch.cat((torch.unsqueeze(d_uf, dim=1), d[n_unfinished:]), dim=0)  # shape: (b, 1, K)
        d = d.squeeze(dim=1)  # shape: (b, K)
        _, max_idx = torch.max(d, dim=1)  # shape: (b,)
        max_idx = max_idx.tolist()
        tags = [tags[b][k] for b, k in enumerate(max_idx)]
        return tags


    def save(self, filepath):
        params = {
            'tag_vocab': self.tag_vocab,
            'args': dict(dropout_rate=self.dropout_rate),
            'state_dict': self.state_dict()
        }
        torch.save(params, filepath)

    @staticmethod
    def load(filepath, device_to_load):
        params = torch.load(filepath, map_location=lambda storage, loc: storage)
        model = BertCRF(params['tag_vocab'], **params['args'])
        model.load_state_dict(params['state_dict'])
        model.to(device_to_load)
        return model