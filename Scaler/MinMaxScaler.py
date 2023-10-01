import torch


class MinMaxScaler:
    def __init__(self):
        self.min = None
        self.max = None

    def fit(self, data):
        tensor_data = torch.tensor(data)

        self.min = torch.min(tensor_data, dim=0)[0]
        self.max = torch.max(tensor_data, dim=0)[0]

    def transform(self, data):
        tensor_data = torch.tensor(data)

        return (tensor_data - self.min) / (self.max - self.min)

    def fit_transform(self, data):
        self.fit(data)

        return self.transform(data)

    def inverse_transform(self, data):
        tensor_data = torch.tensor(data)

        return tensor_data * (self.max - self.min) + self.min
