import numpy as np
from utils.utils import tagseq_to_entityseq


class SampleMetrics(object):
    @classmethod
    def _reading_cost(cls, selected_samples: list) -> float:
        """
        calculate average reading cost of selected samples
        """
        avg_reading_cost = np.mean([len(sample) for sample in selected_samples])
        return float(avg_reading_cost)

    @classmethod
    def _percentage_wrong_selection(cls, y_true: list, y_pred: list) -> float:
        """
        Calculate percentage of unannotation needed samples
        :param y_true: [B, L]
        :param y_pred: [B, L]
        """
        assert len(y_pred) == len(y_true)
        choice_number = len(y_true)
        wrong_number = sum(1 for t, p in zip(y_true, y_pred) if ' '.join(t) != ' '.join(p))
        return wrong_number / choice_number

    @classmethod
    def _annotation_cost(cls, y_true: list, y_pred: list) -> float:
        """
        Calculate the average annotation cost of seleced samples
        """
        assert len(y_true) == len(y_pred)
        annotation_cost = 0
        for t, p in zip(y_true, y_pred):
            t_set, p_set = set(tagseq_to_entityseq(t)), set(tagseq_to_entityseq(p))
            cost = len(t_set | p_set) - len(t_set & p_set)
            annotation_cost += cost
        return annotation_cost / len(y_true)
