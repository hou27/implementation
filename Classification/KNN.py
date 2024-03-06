import torch
from collections import Counter


class KNeighborsClassifier:
    def __init__(self, n_neighbors=3):
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        # 따로 파라미터를 학습하는 것이 아닌 데이터를 저장하는 것이 전부
        self.X_train = torch.tensor(X, dtype=torch.float32)
        self.y_train = torch.tensor(y, dtype=torch.int64)

    def predict(self, x):
        if isinstance(x, torch.Tensor):
            x = x.clone().detach()
        else:
            x = torch.tensor(x, dtype=torch.float32)
        distances = torch.norm(self.X_train - x, dim=1, p=None)  # 데이터 간의 거리를 구함
        knn = distances.topk(self.n_neighbors, largest=False)  # 가장 가까운 k개의 데이터를 구함
        indices = knn.indices
        k_nearest_labels = self.y_train[indices]
        most_common = Counter(k_nearest_labels.numpy()).most_common(
            1
        )  # 가장 많이 등장한 레이블을 구함

        return most_common[0][0]
