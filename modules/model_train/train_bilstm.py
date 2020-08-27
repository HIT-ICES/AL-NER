import sys
sys.path.append("../..")
import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

from modules.model_train.bilstm_crf import BiLSTMCRF
from modules.data_preprocess.data_loader import Preprocessor

ALL_WORD_EMBEDING_PATH = '../../embedding/merge_sgns_bigram_char300.pkl'
EMBEDDING_DIM = 300
TAGS_FILE = '../../datasets/BosonNLP_NER_6C/tags.txt'
COURPUS_FILE = '../../datasets/BosonNLP_NER_6C/'
HIDDEN_DIM = 200
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCH = 20
device = 'cuda:1'


def train():
    embedding_dim = EMBEDDING_DIM
    tags = [line.strip() for line in open(TAGS_FILE, 'r', encoding='utf8').readlines()]
    all_words_embeds = pickle.load(open(ALL_WORD_EMBEDING_PATH, 'rb'))

    vocab = list(all_words_embeds.keys())
    preprocessor = Preprocessor(vocab=vocab, tags=tags)

    word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06), (len(preprocessor.vocab), embedding_dim))

    for word in all_words_embeds:
        word_embeds[preprocessor.word_to_idx[word]] = all_words_embeds[word]

    model = BiLSTMCRF(vocab_size=len(preprocessor.vocab),
                      tag_to_ix=preprocessor.tag_to_idx,
                      embedding_dim=embedding_dim,
                      hidden_dim=HIDDEN_DIM,
                      pre_word_embed=word_embeds,
                      num_rnn_layers=1,
                      )
    model.to(device)

    datapool, eval_xs, eval_ys = preprocessor.load_dataset(COURPUS_FILE, "BosonNLP_NER_6C", 6, 64,
                                                           os.path.join(COURPUS_FILE, "statistics.csv"))
    train_xs, train_ys = datapool.get_annotated_data()
    train_xs = torch.from_numpy(train_xs).int()
    train_ys = torch.from_numpy(train_ys).int()
    train_dl = DataLoader(TensorDataset(train_xs, train_ys), batch_size=64, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0)
    for epoch in range(NUM_EPOCH):
        model.train()
        bar = tqdm(train_dl)
        for bi, (xb, yb) in enumerate(bar):
            model.zero_grad()

            loss = model.loss(xb.to(device), yb.to(device))
            loss.backward()
            optimizer.step()
            bar.set_description("{:2d}/{} loss: {:5.2f}".format(epoch + 1, NUM_EPOCH, loss))
    torch.save(model.state_dict(), '../../model/BosonNLP_NER_6C_full.pth')


if __name__ == '__main__':
    train()
