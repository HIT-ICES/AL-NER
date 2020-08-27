from typing import List

import numpy as np
from abc import ABCMeta, abstractmethod


class ActiveLearningStrategy(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def select_idx(cls, choices_number: int, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_path: List[List[int]] = None, **kwargs) -> np.ndarray:
        """
        probs: [B, L, C]
        scores: [B]
        best_path: [B, L]
        """
        pass


class RandomStrategy(ActiveLearningStrategy):
    @classmethod
    def select_idx(cls, choices_number: int, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_path: List[List[int]] = None, **kwargs) -> np.ndarray:
        """
        Random Select Strategy
        This method you can directly pass candidate_number: int

        .. Note:: Random Select does not require to predict on the unannotated samples!!
        """
        if "candidate_number" in kwargs:
            candidate_number = kwargs["candidate_number"]
        else:
            candidate_number = scores.shape[0]
        return np.random.choice(np.arange(candidate_number), size=choices_number)

class LongStrategy(ActiveLearningStrategy):
    @classmethod
    def select_idx(cls, choices_number: int, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_path: List[List[int]] = None, **kwargs) -> np.ndarray:
        length = np.array([-len(path) for path in best_path])
        return np.argpartition(length, choices_number)[:choices_number]

class LeastConfidenceStrategy(ActiveLearningStrategy):
    @classmethod
    def select_idx(cls, choices_number: int, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_path: List[List[int]] = None, **kwargs) -> np.ndarray:
        """
        Least Confidence Strategy

        """
        assert probs.shape[0] == scores.shape[0] == len(best_path)
        idx = np.argpartition(-scores, choices_number)[:choices_number]
        return idx


class NormalizedLeastConfidenceStrategy(ActiveLearningStrategy):
    @classmethod
    def select_idx(cls, choices_number: int, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_path: List[List[int]] = None, **kwargs) -> np.ndarray:
        """
        Normalized Least Confidence Strategy
        """
        assert probs.shape[0] == scores.shape[0] == len(best_path)
        normalized_frac = np.array([len(path) for path in best_path])
        scores = scores / normalized_frac
        idx = np.argpartition(-scores, choices_number)[:choices_number]
        return idx


class LeastTokenProbabilityStrategy(ActiveLearningStrategy):
    @classmethod
    def select_idx(cls, choices_number: int, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_path: List[List[int]] = None, **kwargs) -> np.ndarray:
        """
        Least Token Probability Strategy
        """
        assert probs.shape[0] == scores.shape[0] == len(best_path)
        ltp_scores = []
        for prob, path in zip(probs, best_path):
            prob = np.take(prob, path)
            ltp_scores.append(np.min(prob))
        idx = np.argpartition(ltp_scores, choices_number)[:choices_number]
        return idx


class MinimumTokenProbabilityStrategy(ActiveLearningStrategy):
    @classmethod
    def select_idx(cls, choices_number: int, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_path: List[List[int]] = None, **kwargs) -> np.ndarray:
        """
        Minimum Token Probability Strategy
        """
        assert probs.shape[0] == scores.shape[0] == len(best_path)
        mtp_socres = []
        for prob, path in zip(probs, best_path):
            prob = prob[:len(path)]
            prob -= np.max(prob)
            prob = np.exp(prob) / np.sum(np.exp(prob))
            mtp_socres.append(np.min(np.max(prob[:len(path)], axis=1)))
        idx = np.argpartition(mtp_socres, choices_number)[:choices_number]
        return idx


class MaximumTokenEntropyStrategy(ActiveLearningStrategy):
    """
    TTE
    """

    @classmethod
    def select_idx(cls, choices_number: int, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_path: List[List[int]] = None, **kwargs) -> np.ndarray:
        """
        Maximum Token Entropy
        """
        assert probs.shape[0] == scores.shape[0] == len(best_path)
        mte_socres = []
        for prob, path in zip(probs, best_path):
            prob = prob[:len(path)]
            prob -= np.max(prob)
            prob_softmax = np.exp(prob) / np.sum(np.exp(prob))
            mte_socres.append(np.sum(prob_softmax * np.log(prob_softmax)))
        idx = np.argpartition(mte_socres, choices_number)[:choices_number]
        return idx


class TokenEntropyStrategy(ActiveLearningStrategy):
    """
    TTE
    """

    @classmethod
    def select_idx(cls, choices_number: int, probs: np.ndarray = None, scores: np.ndarray = None,
                   best_path: List[List[int]] = None, **kwargs) -> np.ndarray:
        """
        Maximum Token Entropy
        """
        assert probs.shape[0] == scores.shape[0] == len(best_path)
        mte_socres = []
        for prob, path in zip(probs, best_path):
            prob = prob[:len(path)]
            prob -= np.max(prob)
            prob_softmax = np.exp(prob) / np.sum(np.exp(prob))
            mte_socres.append(np.mean(prob_softmax * np.log(prob_softmax)))
        idx = np.argpartition(mte_socres, choices_number)[:choices_number]
        return idx
