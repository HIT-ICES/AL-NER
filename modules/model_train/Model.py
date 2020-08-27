import torch.nn as nn


class Model(object):
    def __init__(self, model: nn.Module):
        self.model = model

    def train(self, config_file, x=None, y=None):
        pass

    def evaluate(self, config_file, x=None, y=None):
        pass

    def predict(self, config_file, x=None):
        pass