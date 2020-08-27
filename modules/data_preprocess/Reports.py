from typing import Callable, Any

import pandas as pd


class DatasetStatistics(dict):
    def statistics(self, texts, labels, func: Callable[[Any, Any], dict]) -> None:
        pass
