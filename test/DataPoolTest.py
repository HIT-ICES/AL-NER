import unittest
import numpy as np
import random
from modules.data_preprocess.DataPool import DataPool


class DataPoolTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.annotated_texts = [''.join(random.choices(list("abcdefghijklmn"), k=10)) for i in range(10)]
        self.annotated_labels = [random.choices(list("BIO"), k=len(text)) for text in self.annotated_texts]
        self.unannotated_texts = [''.join(random.choices(list("abcdefghijklmn"), k=10)) for i in range(20)]
        self.unannotated_labels = [random.choices(list("BIO"), k=len(text)) for text in self.unannotated_texts]
        self.data_pool = DataPool(self.annotated_texts, self.annotated_labels, self.unannotated_texts, self.unannotated_labels)

    def test_get_annotated_data(self):
        obtained_texts, obtained_labels = self.data_pool.get_annotated_data()
        self.assertListEqual(self.annotated_texts, obtained_texts.tolist())
        self.assertListEqual(self.annotated_labels, obtained_labels.tolist())

    def test_get_unannotated_text(self):
        obtained_texts, obtained_labels = self.data_pool.get_unannotated_data()
        self.assertListEqual(self.unannotated_texts, obtained_texts.tolist())
        self.assertListEqual(self.unannotated_labels, obtained_labels.tolist())

    def test_append_annotated_update(self):
        texts =[''.join(random.choices(list("abcdefghijklmn"), k=10)) for i in range(5)]
        labels = [random.choices(list("BIO"), k=len(text)) for text in texts]
        expected_texts, expected_labels = self.data_pool.get_annotated_data()
        expected_texts = np.concatenate((expected_texts, np.array(texts)))
        expected_labels = np.concatenate((expected_labels, np.array(labels)))
        self.data_pool.update(mode="append_annotated", annotated_texts=texts, annotated_labels=labels)
        actucal_texts, actucal_labels = self.data_pool.get_annotated_data()
        self.assertListEqual(expected_texts.tolist(), actucal_texts.tolist())

    def test_append_unannotated_update_1(self):
        texts =[''.join(random.choices(list("abcdefghijklmn"), k=10)) for i in range(5)]
        expected_texts, expected_labels = self.data_pool.get_unannotated_data()
        expected_texts = np.concatenate((expected_texts, np.array(texts)))
        labels = [['O' for j in range(len(i))] for i in texts]
        expected_labels = expected_labels.tolist()
        expected_labels.extend(labels)
        expected_labels = np.array(expected_labels)
        self.data_pool.update(mode="append_unannotated", unannotated_texts=texts, unannotated_labels=None)
        actucal_texts, actucal_labels = self.data_pool.get_unannotated_data()
        self.assertListEqual(expected_texts.tolist(), actucal_texts.tolist())

    def test_append_unannotated_update_2(self):
        texts =[''.join(random.choices(list("abcdefghijklmn"), k=10)) for i in range(5)]
        labels = [random.choices(list("BIO"), k=len(text)) for text in texts]
        expected_texts, expected_labels = self.data_pool.get_unannotated_data()
        expected_texts = np.concatenate((expected_texts, np.array(texts)))
        expected_labels = np.concatenate((expected_labels, np.array(labels)))
        self.data_pool.update(mode="append_unannotated", unannotated_texts=texts, unannotated_labels=labels)
        actucal_texts, actucal_labels = self.data_pool.get_unannotated_data()
        self.assertListEqual(expected_texts.tolist(), actucal_texts.tolist())

    def test_internal_exchange_u2a(self):
        annotated_texts, annotated_labels = self.data_pool.get_annotated_data()
        unannotated_texts, unannotated_labels = self.data_pool.get_unannotated_data()
        annotated_texts, annotated_labels = annotated_texts.tolist(), annotated_labels.tolist()
        unannotated_texts, unannotated_labels = unannotated_texts.tolist(), unannotated_labels.tolist()
        selected_idx = [1, 3, 0]
        for idx in selected_idx:
            annotated_texts.append(unannotated_texts[idx])
            annotated_labels.append(unannotated_labels[idx])
        unannotated_texts = [x for idx, x in enumerate(unannotated_texts) if idx not in selected_idx]
        unannotated_labels = [x for idx, x in enumerate(unannotated_labels) if idx not in selected_idx]
        self.data_pool.update(mode="internal_exchange_u2a", selected_idx=selected_idx)
        actucal_anno_texts, actucal_anno_labels = self.data_pool.get_annotated_data()
        actucal_unan_texts, actucal_unan_labels = self.data_pool.get_unannotated_data()
        self.assertListEqual(annotated_texts, actucal_anno_texts.tolist())
        self.assertListEqual(unannotated_texts, actucal_unan_texts.tolist())
        self.assertListEqual(annotated_labels, actucal_anno_labels.tolist())
        self.assertListEqual(unannotated_labels, actucal_unan_labels.tolist())

    def test_a2u_2(self):
        data_pool = DataPool(annotated_texts=self.annotated_texts, annotated_labels=self.annotated_labels,
                             unannotated_texts=[[]], unannotated_labels=[[]])
        data_pool.update(mode="internal_exchange_a2u", selected_idx=np.array([0, 1]))


if __name__ == '__main__':
    unittest.main()
