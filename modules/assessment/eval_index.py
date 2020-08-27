import sys
sys.path.append("../..")

from seqeval import metrics as entity_metrics
from sklearn import metrics as sk_metrics

# AL-NER-DEMO Modules
from utils.logger import Logger
from utils.utils import flatten_lists


class EvaluationIndex(object):

    def __init__(self, logger):
        self.logger = logger

    def entity_level_f1(self, y_true, y_pred, digits=2, return_report=False, average="micro"):
        """
        entity-level-f1
        :params golden_tags Tags given manually
        :params predict_tags Prediction tags given by the model
        :return f1 score
        """
        assert len(y_true) == len(y_pred)
        score = entity_metrics.f1_score(y_true, y_pred, average=average)
        report = entity_metrics.classification_report(y_true, y_pred, digits=digits)
        self.logger.info(f"Classification report(Entity level):\n{report}")
        if return_report:
            return score, report
        return score

    def sentence_level_accuracy(self, y_true, y_pred):
        """
        sentence-level accuracy:
        :param y_true: golden_tags given manually
        :param y_pred: predicted_tags given by the model
        """
        assert len(y_true) == len(y_pred)
        y_true, y_pred = [' '.join(y) for y in y_true], [' '.join(y) for y in y_pred]
        score = sk_metrics.accuracy_score(y_true, y_pred)
        self.logger.info(f"Sentence-level Accuracy: {score}")
        return score
