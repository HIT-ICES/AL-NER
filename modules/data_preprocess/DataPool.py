import numpy as np
from typing import Tuple


class DataPool(object):
    """
    DataPool is used to store all datas, including annotated and unannotated data.
    DataPool also provide a unified data acquisition interface.
    """

    def __init__(self, annotated_texts: list, annotated_labels: list,
                 unannotated_texts: list, unannotated_labels: list) -> None:
        """
        initialize DataPool object
        :param annotated_texts: annotated samples' text, must equal to annotated_labels
        :param annotated_labels: annotated samples' label, must equal to annotated_texts
        :param unannotated_texts: unannotated samples' text, must equal to unannotated_labels if unannotated labels is not None
        :param unannotated_labels: unannotated samples' label
        """
        if len(annotated_texts) != len(annotated_labels):
            raise ValueError(f"unequal of texts-{len(annotated_texts)} and labels-{len(annotated_labels)}")

        if unannotated_labels is not None:  
            if len(unannotated_texts) != len(unannotated_labels):
                raise ValueError(f"unequal of texts-{len(unannotated_texts)} and labels-{len(annotated_labels)}")
        else:
            unannotated_labels = [["O" for j in range(len(i))] for i in unannotated_texts]    # make sure they have same length

        self.annotated_texts = np.array(annotated_texts)
        self.annotated_labels = np.array(annotated_labels)
        self.unannotated_texts = np.array(unannotated_texts)
        self.unannotated_labels = np.array(unannotated_labels)

    def get_annotated_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        obtain all annotated data
        :return: annotated_texts and corresponding labels
        """
        return self.annotated_texts, self.annotated_labels

    def get_unannotated_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        obtain all unannotated data
        :return: unannotated_texts and corresponding laebls
        """
        return self.unannotated_texts, self.unannotated_labels

    def update(self, mode: str, annotated_texts=None, annotated_labels=None,
               unannotated_texts=None, unannotated_labels=None, selected_idx=None):
        """
        Update the data in the data pool.

        .. Example:
        >>> texts =[''.join(random.choices(list("abcdefghijklmn"), k=10)) for i in range(5)]
        >>> labels = [random.choices(list("BIO"), k=len(text)) for text in texts]
        >>> data_pool.update(mode="append_annotated", annotated_texts=texts, annotated_labels=labels)


        :param mode: operation mode one of ("append_annotated", "append_unannotated", "replace_annotated",
            "replace_unannotated", "internal_exchange_a2u", "internal_exchange_u2a")
        :param annotated_texts: annotated samples' text, only works when `mode` is "append_annotated" and
            "replace_annotated"
        :param annotated_labels: annotated samples' label, only works when `mode` is "append_annotated",
            "replace_annotated" and "internal_exchange_a2u"
        :param unannotated_texts: unannotated samples' text, only works when `mode` is "append_unannotated" and
            "replace_unannotated"
        :param unannotated_labels: annotated samples' label, only works when `mode` is "append_unannotated",
            "replace_unannotated" and "internal_exchange_u2a"
        :param selected_idx: selected in-pool samples' idx, only works when `mode` is "internal_exchange_u2a" and
            "internal_exchange_a2u"
        :raise ValueError: when params length are not correct.
        """
        predefined_modes = ("append_annotated", "append_unannotated", "replace_annotated",
                            "replace_unannotated", "internal_exchange_a2u", "internal_exchange_u2a")
        if mode not in predefined_modes:
            raise ValueError(f"mode value must be one of {predefined_modes}")

        if mode == "append_annotated" or mode == "replace_annotated":
            if annotated_texts is None or annotated_labels is None:
                raise ValueError(f"In {mode} mode, annotated_texts and annotated_labels cannot be None")
            if len(annotated_texts) != len(annotated_labels):    # make sure they have same length
                raise ValueError("annotated_labels and annotated_texts must have same length")
            self.annotated_texts = np.concatenate((self.annotated_texts, np.array(annotated_texts))) \
                if mode == "append_annotated" else np.array(annotated_texts)

            self.annotated_labels = np.concatenate((self.annotated_labels, np.array(annotated_labels))) \
                if mode == "append_annotated" else np.array(annotated_labels)

        elif mode == "append_unannotated" or mode == "replace_unannotated":
            if unannotated_texts is None:
                raise ValueError(f"In {mode} mode, unannotated_texts cannot be None")
            if unannotated_labels is None:
                unannotated_labels = [["O" for j in range(len(i))] for i in unannotated_texts]
            if len(unannotated_texts) != len(unannotated_labels):
                raise ValueError("unannotated_labels and unannotated_texts must have same length")

            self.unannotated_texts = np.concatenate((self.unannotated_texts, np.array(unannotated_texts))) \
                if mode == "append_unannotated" else np.array(unannotated_texts)
            self.unannotated_labels = np.concatenate((self.unannotated_labels, np.array(unannotated_labels))) \
                if mode == "append_unannotated" else np.array(unannotated_labels)

        elif mode == "internal_exchange_a2u":
            # move samples from annotated database to unannotated database
            if self.unannotated_texts.shape[-1] != 0:
                self.unannotated_texts = np.concatenate((self.unannotated_texts, self.annotated_texts[selected_idx]))
                self.unannotated_labels = np.concatenate((self.unannotated_labels, self.annotated_labels[selected_idx]))
            else:
                self.unannotated_texts = self.annotated_texts[selected_idx]
                self.unannotated_labels = self.annotated_labels[selected_idx]
            # delete selected samples from annotated database
            self.annotated_texts = np.delete(self.annotated_texts, selected_idx, axis=0)
            self.annotated_labels = np.delete(self.annotated_labels, selected_idx, axis=0)

        elif mode == "internal_exchange_u2a":
            if unannotated_labels is not None:
                if len(unannotated_labels) != len(selected_idx):
                    raise ValueError(f"In {mode} mode, if unannotated_labels is not None, they must"
                                     f" have the same length as the selected_idx")
                if self.annotated_labels.shape[-1] != 0:
                    self.annotated_labels = np.concatenate((self.annotated_labels, np.array(unannotated_labels)))
                else:
                    self.annotated_labels = np.array(unannotated_labels)
            else:
                if self.annotated_labels.shape[-1] != 0:
                    self.annotated_labels = np.concatenate(
                        (self.annotated_labels, self.unannotated_labels[selected_idx]))
                else:
                    self.annotated_labels = self.unannotated_labels[selected_idx]
            if self.annotated_labels.shape[-1] != 0:
                self.annotated_texts = np.concatenate((self.annotated_texts, self.unannotated_texts[selected_idx]))
            else:
                self.annotated_texts = self.unannotated_texts[selected_idx]
            # delete samples form unannotated database
            self.unannotated_texts = np.delete(self.unannotated_texts, selected_idx, axis=0)
            self.unannotated_labels = np.delete(self.unannotated_labels, selected_idx, axis=0)

    def get_total_number(self) -> int:
        """
        Get total number of samples.
        """
        return len(self.annotated_texts) + len(self.unannotated_texts)

