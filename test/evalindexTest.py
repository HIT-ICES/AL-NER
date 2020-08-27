import sys
sys.path.append("..")
import unittest
from modules.assessment.eval_index import EvaluationIndex

from utils.logger import Logger


class MyTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.evaluator = EvaluationIndex(Logger(__name__))

    def test_sentence_level_accuracy(self):
        y_true = [["a", "b"], ["a", "c"]]
        y_pred = [["a", "b"], ["a", "a"]]
        score = self.evaluator.sentence_level_accuracy(y_true, y_pred)
        self.assertEqual(0.5, score)


if __name__ == '__main__':
    unittest.main()
