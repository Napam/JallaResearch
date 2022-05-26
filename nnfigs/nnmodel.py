import torch
import pandas as pd
from torch import nn

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.line = nn.Linear(in_features=2, out_features=1)

    def forward(self, X: torch.Tensor):
        return torch.sigmoid(self.line(X))

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        out = self.forward(X)
        print(out.shape)
        print(y.shape)
        print((out - y)**2)

df = pd.read_csv("datasets/apples_and_oranges.csv")
model = Model()

X = torch.tensor(df[["weight", "height"]].values, dtype=torch.float32)
y = torch.tensor(df["class"].map({"apple": 0, "orange": 1}).values, dtype=torch.float32)

model.fit(X, y)
        