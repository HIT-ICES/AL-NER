import os
import sys
sys.path.append("..")

import torch
import pickle
import numpy as np
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim

# AL-NER-DEMO Modules
from utils.logger import Logger
from utils.utils import vec_to_tags
from core.pipeline import Pipeline
from modules.model_train.bilstm_crf import BiLSTMCRF
from modules.data_preprocess.data_loader import Preprocessor
from modules.data_preprocess.DataPool import DataPool
from modules.assessment.eval_index import EvaluationIndex
from modules.select_strategy.ALStrategy import *
from modules.assessment.sample_metrics import SampleMetrics


class Word2VecBiLSTMCRFALPipeline(Pipeline):
    """
    Word2vec(pre-trained) + BiLSTM + CRF + AL
    ===============================
    """

    def __init__(self):
        self.preprocessor = Preprocessor(vocab=[], tags=[])
        self.datapool = None
        # TODO: Complete
        self.strategy = {
            "RANDOM": RandomStrategy,
            "LC": LeastConfidenceStrategy,
            "NLC": NormalizedLeastConfidenceStrategy,
            "LTP": LeastTokenProbabilityStrategy,
            "MTP": MinimumTokenProbabilityStrategy,
            "MTE": MaximumTokenEntropyStrategy,
            "LONG": LongStrategy,
            "TE": TokenEntropyStrategy,
        }
        super(Word2VecBiLSTMCRFALPipeline, self).__init__()

    def word_embedding(self):
        """
        Step 01
        Embedding with word2vec.
        """
        self.logger.info("Step01 Begin: word embedding.\n")

        all_word_embedding_path = self.config.param("WORD2VEC", "all_word_embedding_path", type="filepath")
        courpus_file = self.config.param("WORD2VEC", "courpus_file", type="dirpath")
        courpus_name = self.config.param("WORD2VEC", "courpus_name", type="string")
        choose_fraction = self.config.param("WORD2VEC", "choose_fraction", type="float")
        embedding_dim = self.config.param("WORD2VEC", "embedding_dim", type="int")
        entity_type = self.config.param("WORD2VEC", "entity_type", type="int")
        max_seq_len = self.config.param("WORD2VEC", "max_seq_len", type="int")
        tags_file = self.config.param("WORD2VEC", "tags_file", type="filepath")

        self.tags = [line.strip() for line in open(tags_file, 'r', encoding='utf8').readlines()]
        self.labels = [label for label in self.tags if label not in ['O', '[PAD]', '[CLS]', '[SEP]', 'X']]
        all_words_embeds = pickle.load(open(all_word_embedding_path, 'rb'))
        vocab = list(all_words_embeds.keys())
        self.preprocessor = Preprocessor(vocab=vocab, tags=self.tags)

        self.word_embeds = np.random.uniform(-np.sqrt(0.06), np.sqrt(0.06),
                                             (len(self.preprocessor.vocab), embedding_dim))

        for word in all_words_embeds:
            self.word_embeds[self.preprocessor.word_to_idx[word]] = all_words_embeds[word]

        self.datapool, self.eval_xs, self.eval_ys = self.preprocessor.load_dataset_init(courpus_file, courpus_name,
                                                                                        entity_type, choose_fraction,
                                                                                        max_seq_len,
                                                                                        os.path.join(courpus_file,
                                                                                                     "statistics.csv"))

        self.eval_ys = vec_to_tags(self.tags, self.eval_ys, max_seq_len)
        self.logger.info("Step01 Finish: word embedding.\n")
        return

    def build_bilstm_crf(self):
        """
        Step 02
        Build BiLSTM+CRF Model.
        """
        self.logger.info("Step02 Begin: build bilstm crf.\n")

        batch_size = self.config.param("BiLSTMCRF", "batch_size", type="int")
        device = self.config.param("BiLSTMCRF", "device", type="string")
        embedding_dim = self.config.param("BiLSTMCRF", "embedding_dim", type="int")
        hidden_dim = self.config.param("BiLSTMCRF", "hidden_dim", type="int")
        learning_rate = self.config.param("BiLSTMCRF", "learning_rate", type="float")
        model_path_prefix = self.config.param("BiLSTMCRF", "model_path_prefix", type="string")
        num_rnn_layers = self.config.param("BiLSTMCRF", "num_rnn_layers", type="int")
        num_epoch = self.config.param("BiLSTMCRF", "num_epoch", type="int")

        pre_step_name = 'word_embedding'

        train_xs, train_ys = self.datapool.get_annotated_data()
        train_xs = torch.from_numpy(train_xs).int()
        train_ys = torch.from_numpy(train_ys).int()
        train_dl = DataLoader(TensorDataset(train_xs, train_ys), batch_size, shuffle=True)

        self.model = BiLSTMCRF(vocab_size=len(self.preprocessor.vocab),
                    tag_to_ix=self.preprocessor.tag_to_idx,
                    embedding_dim=embedding_dim,
                    hidden_dim=hidden_dim,
                    pre_word_embed=self.word_embeds,
                    num_rnn_layers=num_rnn_layers,
                    )
        self.model.to(device)

        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=0)
        info = ""
        for epoch in range(num_epoch):
            self.model.train()
            bar = tqdm(train_dl)
            for bi, (xb, yb) in enumerate(bar):
                self.model.zero_grad()
                loss = self.model.loss(xb.to(device), yb.to(device))
                loss.backward()
                optimizer.step()
                bar.set_description(f"{epoch + 1:2d}/{num_epoch} loss: {loss:5.2f}")
            info += f"{epoch + 1:2d}/{num_epoch} loss: {loss:5.2f}\n"
        self.logger.info(f"{info}")
        torch.save(self.model.state_dict(), model_path_prefix + ".pth")
        self.logger.info("Step02 Finish: bilstm crf.\n")

        return

    def predict_eval(self):
        """
        Step 03
        Use training model to predict and evaluate
        entity-level-F1
        sentence-level-accuracy
        """
        self.logger.info("Step03 Begin: Predicting and evaluation.\n")
        device = self.config.param("BiLSTMCRF", "device", type="string")
        max_seq_len = self.config.param("WORD2VEC", "max_seq_len", type="int")
        entity_digits = self.config.param("ENTITYLEVELF1", "digits", type="int")
        entity_return_report = self.config.param("ENTITYLEVELF1", "return_report", type="boolean")
        entity_average = self.config.param("ENTITYLEVELF1", "average", type="string")

        self.model.eval()
        self.eval_dl = torch.from_numpy(np.array(self.eval_xs)).int().to(device)
        scores, tag_seq, probs = None, None, None
        with torch.no_grad():

            scores, tag_seq, probs = self.model(self.eval_dl)
        tag_seq = vec_to_tags(self.tags, tag_seq, max_seq_len)

        eval = EvaluationIndex(self.logger)

        if entity_return_report:
            entity_f1_score, entity_return_report = eval.entity_level_f1(self.eval_ys, tag_seq,
                                                                         entity_digits, entity_return_report,
                                                                         entity_average)
            print(f"Classification report(Entity level):\n{entity_return_report}")
        else:
            entity_f1_score = eval.entity_level_f1(self.eval_ys, tag_seq, entity_digits,
                                                   entity_return_report, entity_average)

        self.logger.info(f"Entity-level F1: {entity_f1_score}")

        sentence_ac_score = eval.sentence_level_accuracy(self.eval_ys, tag_seq)
        print(f"Sentence-level Accuracy: {sentence_ac_score}")

        self.logger.info("Step03 Finish: Predicting and evaluation.\n")

        return entity_f1_score,sentence_ac_score

    def eval(self):
        """
        Step 03
        Use training model to predict and evaluate
        entity-level-F1
        sentence-level-accuracy
        """
        self.logger.info("Step03 Begin: Predicting and evaluation.\n")
        device = self.config.param("BiLSTMCRF", "device", type="string")
        max_seq_len = self.config.param("WORD2VEC", "max_seq_len", type="int")

        unannotated_texts, unannotated_labels = self.datapool.get_unannotated_data()
        unannotated_texts = torch.from_numpy(unannotated_texts).int()
        unannotated_labels = torch.from_numpy(unannotated_labels).int()
        eval_dl = DataLoader(TensorDataset(unannotated_texts, unannotated_labels), 64, shuffle=False)
        self.model.eval()
        scores, tag_seq_l, probs, tag_seq_str = [], [], [], []
        with torch.no_grad():
            bar = tqdm(eval_dl)
            for bi,(xs,ys) in enumerate(bar):
                score, tag_seq, prob = self.model(xs.to(device))
                score, prob = score.cpu().detach().numpy(), prob.cpu().detach().numpy()
                tag_seq_l.extend(tag_seq)
                scores.extend(score.tolist())
                probs.extend(prob.tolist())
                tag_seq_str.extend(vec_to_tags(self.tags, tag_seq, max_seq_len))
        scores = np.array(scores)
        probs = np.array(probs)
        return scores, tag_seq_l, probs, tag_seq_str

    def active_learning(self):
        self.logger.info("Begin active_learning.")
        strategy = self.config.param("ActiveStrategy", "strategy", type="string")
        strategy_name = strategy.lower()
        stop_echo = self.config.param("ActiveStrategy", "stop_echo", type="int")
        query_batch_fraction = self.config.param("ActiveStrategy", "query_batch_fraction", type="float")
        max_seq_len = self.config.param("WORD2VEC", "max_seq_len", type="int")
        choice_number = int(self.datapool.get_total_number() * query_batch_fraction)
        strategy = self.strategy[strategy]
        for i in range(0, stop_echo):
            self.logger.info(
                f"[No. {i + 1}/{stop_echo}] ActiveStrategy:{strategy}, BatchFraction: {query_batch_fraction}\n")
            self.build_bilstm_crf()
            entity_f1_score,sentence_ac_score = self.predict_eval()
            scores, tag_seq, probs, tag_seq_str = self.eval()
            _, unannotated_labels = self.datapool.get_unannotated_data()
            idx = strategy.select_idx(choices_number=choice_number, probs=probs, scores=scores, best_path=tag_seq)
            selected_samples = unannotated_labels[idx]
            selected_samples = vec_to_tags(self.tags, selected_samples.tolist(), max_seq_len)
            tag_seq_str = [tag_seq_str[id] for id in idx]
            # update datapool
            self.datapool.update(mode="internal_exchange_u2a", selected_idx=idx)
            _reading_cost = SampleMetrics._reading_cost(selected_samples)
            self.logger.info(f"Reading Cost is {_reading_cost}")
            _annotation_cost = SampleMetrics._annotation_cost(selected_samples, tag_seq_str)
            self.logger.info(f"Annotation Cost is {_annotation_cost}")
            _wrong_select = SampleMetrics._percentage_wrong_selection(selected_samples, tag_seq_str)
            self.logger.info(f"Wrong Selected percentage: {_wrong_select}")
            self.logger.info(f"{strategy_name},{i},{entity_f1_score},{sentence_ac_score},{_reading_cost},{_annotation_cost},{_wrong_select}")
            del self.model
            torch.cuda.empty_cache()

    @property
    def tasks(self):
        return [
            self.word_embedding,
            self.active_learning,
        ]


if __name__ == '__main__':
    Word2VecBiLSTMCRFALPipeline()

