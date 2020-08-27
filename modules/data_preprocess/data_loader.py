from .DataPool import DataPool
import os
import numpy as np

class Preprocessor(object):
    def __init__(self, vocab, tags):
        self.vocab = vocab
        self.vocab.insert(0, "[PAD]")
        if '[CLS]' not in self.vocab:
            self.vocab.append('[CLS]')
        if '[SEP]' not in self.vocab:
            self.vocab.append('[SEP]')
        self.vocab.append("[OOV]")
        self.tags = tags
        self.PAD_IDX = 0
        self.OOV_IDX = len(self.vocab) - 1
        self.word_to_idx = {key: idx for idx, key in enumerate(self.vocab)}
        self.tag_to_idx = {key: idx for idx, key in enumerate(self.tags)}

    def _entity_number(self, labels):
        """
        Calculate how many enities in a sentence.
        """
        return sum([1 for label in labels if label.startswith('B-')])

    def update_statistics(self, statistics: dict, annotated_label: list) -> dict:
        statistics["#S"] += 1
        statistics["#T"] += len(annotated_label)
        statistics["ASL"] += len(annotated_label)
        positive_tags_num = len([label for label in annotated_label if label != 'O'])
        statistics["AEL"] += positive_tags_num
        statistics["%PT"] += positive_tags_num
        entity_number = self._entity_number(annotated_label)
        statistics["TE"] += entity_number
        if entity_number >= 1:
            statistics['%AC'] += 1
        if entity_number >= 2:
            statistics['%DAC'] += 1
        return statistics

    def formatter_statistics(slef, statistics: dict) -> str:
        statistics["ASL"] /= statistics["#S"]
        statistics["AEL"] /= statistics["TE"]
        statistics["%PT"] /= statistics["#T"]
        statistics["%AC"] /= statistics["#S"]
        statistics["%DAC"] /= statistics["#S"]
        statistics.pop("TE")
        return '\t'.join([f'{key}:{value}' for key, value in statistics.items()])

    def load_dataset(self, corpus_dir, name: str, entity_type: int, max_seq_len=64,
                     statistics_report: str = None) -> tuple:
        """Loads dataset from corpus_dir and returns a tuple
        :param corpus_dir: directory of corpus
        :param name: dataset name
        :param entity_type: the number of the entity type
        :param statistics_report: write statistics information to file or not
        :return: (DataPool, evaluation_xs, evaluation_ys)
        """
        train_path = os.path.join(corpus_dir, "train.txt")
        test_path = os.path.join(corpus_dir, "test.txt")

        annotated_labels, train_statistics, annotated_texts = self._load_from_file(entity_type, name + "-train",
                                                                                   train_path, max_seq_len)
        eval_labels, eval_statistics, eval_texts = self._load_from_file(entity_type, name + "-test", test_path,
                                                                        max_seq_len)
        datapool = DataPool(annotated_texts=annotated_texts, annotated_labels=annotated_labels,
                            unannotated_texts=[], unannotated_labels=[])
        # append statistics report to file.
        # if statistics_report is not None:
            # with open(statistics_report, 'a', encoding='utf8') as wf:
                # wf.write(f'{self.formatter_statistics(train_statistics)}\n')
                # wf.write(f'{self.formatter_statistics(eval_statistics)}\n')
        return datapool, eval_texts, eval_labels

    def _load_from_file(self, entity_type, name, train_path, max_seq_len):
        lines = open(train_path, 'r', encoding='utf8').readlines()
        texts, labels = [], []
        text, label = [], []
        statistics = {"corpus_name": name, "#E": entity_type, "#S": 0, "#T": 0,
                      "ASL": 0, "AEL": 0, "%PT": 0, "%AC": 0, "%DAC": 0, "TE": 0}
        for line in lines:
            if len(line) < 2:
                if len(text) < 2:  # To avoid empty lines
                    text, label = [], []
                    continue
                texts.append(self.sentences_to_vec(text, max_seq_len))
                labels.append(self.tags_to_vec(label, max_seq_len))
                statistics = self.update_statistics(statistics, label)
                text, label = [], []
                continue
            char, tag = line.strip().split()
            text.append(char)
            label.append(tag)
        return labels, statistics, texts

    def sentences_to_vec(self, sentence, max_seq_len=64):
        vec = [self.word_to_idx.get(word, self.OOV_IDX) for word in sentence[:max_seq_len]]
        return vec + [0] * (max_seq_len - len(vec))

    def tags_to_vec(self, tags, max_seq_len=64):
        vec = [self.tag_to_idx.get(tag) for tag in tags[:max_seq_len]]
        return vec + [0] * (max_seq_len - len(vec))

    def load_dataset_init(self, corpus_dir, name: str, entity_type: int, choose_fraction: float, max_seq_len=64,
                     statistics_report: str = None) -> tuple:
        """Loads dataset from corpus_dir and returns a tuple
        :param corpus_dir: directory of corpus
        :param name: dataset name
        :param entity_type: the number of the entity type
        :param statistics_report: write statistics information to file or not
        :return: (DataPool, evaluation_xs, evaluation_ys)
        """
        train_path = os.path.join(corpus_dir, "train.txt")
        test_path = os.path.join(corpus_dir, "test.txt")

        labels, train_statistics, texts = self._load_from_file(entity_type, name + "-train",
                                                                                   train_path, max_seq_len)
        eval_labels, eval_statistics, eval_texts = self._load_from_file(entity_type, name + "-test", test_path,
                                                                        max_seq_len)
        annotated_idx = np.random.choice(np.arange(len(labels)), size=int(choose_fraction*len(labels)))
        un_annotated_idx = set(range(len(labels))) - set(annotated_idx)

        datapool = DataPool(annotated_texts=[texts[idx] for idx in annotated_idx],
                            annotated_labels=[labels[idx] for idx in annotated_idx],
                            unannotated_texts=[texts[idx] for idx in un_annotated_idx],
                            unannotated_labels=[labels[idx] for idx in un_annotated_idx])

        # append statistics report to file.
        return datapool, eval_texts, eval_labels
