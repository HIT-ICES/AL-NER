import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from .crf import CRF


class BiLSTMCRF(nn.Module):
    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, pre_word_embed=None, num_rnn_layers=1):
        super(BiLSTMCRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix

        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        if pre_word_embed is not None:
            self.pre_word_embed = True
            self.word_embedding.weight = nn.Parameter(torch.FloatTensor(pre_word_embed))

        self.bilstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=num_rnn_layers,
                              bidirectional=True, batch_first=True)
        self.crf = CRF(hidden_dim, self.tag_to_ix)

    def __build_features(self, sentences):
        masks = sentences.gt(0)
        embeds = self.word_embedding(sentences.long())
        seq_length = masks.sum(1)
        sorted_seq_length, perm_idx = seq_length.sort(descending=True)
        embeds = embeds[perm_idx, :]

        pack_sequence = pack_padded_sequence(embeds, lengths=sorted_seq_length, batch_first=True)
        packed_output, _ = self.bilstm(pack_sequence)
        lstm_out, _ = pad_packed_sequence(packed_output, batch_first=True)
        _, unperm_idx = perm_idx.sort()
        lstm_out = lstm_out[unperm_idx, :]
        return lstm_out, masks

    def loss(self, xs, tags):
        features, masks = self.__build_features(xs)
        loss = self.crf.loss(features, tags, masks=masks)
        return loss

    def forward(self, xs):
        # Get the emission scores from the BiLSTM
        features, masks = self.__build_features(xs)
        scores, tag_seq, probs = self.crf(features, masks)
        return scores, tag_seq, probs
