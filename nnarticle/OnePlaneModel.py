from typing import Optional
import torch
import pandas as pd
from torch import nn
from torch import optim
from matplotlib import pyplot as plt

from utils import get_lims
from LineWithDirection import plotHyperplane

def mse(y_: torch.Tensor, y: torch.Tensor):
    assert y_.shape == y.shape
    return ((y_ - y) ** 2).mean()

def accuracy(y_: torch.Tensor, y: torch.Tensor):
    y_, y = y_.detach(), y.detach()
    return ((y_ > 0.5).float() == y).sum() / len(y)

class OnePlaneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.plane = nn.Linear(in_features=2, out_features=1)
        self.X_mean: Optional[torch.Tensor] = None
        self.X_std: Optional[torch.Tensor] = None

    def forward(self, X: torch.Tensor):
        return torch.sigmoid(self.plane(X))


    def fit(self, X: torch.Tensor, y: torch.Tensor):
        optimizer = optim.Adam(self.parameters(), lr=1e-2)
        criterion = mse

        losses = []
        for i in range(10000):
            y_ = self.forward(X)
            loss = criterion(y_, y)
            print(f"Loss: {loss.item():<25} Accuracy: {accuracy(y_, y).item()}")
            if losses and torch.allclose(losses[-1], loss, atol=1e-4):
                print("Achieved satisfactory loss convergence")
                break
            losses.append(loss.detach())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            print("Hit maximum iteration")

    def plot(self, X: torch.Tensor, y: torch.Tensor, X_mean: torch.Tensor, X_std: torch.Tensor):
        X = X * X_std + X_mean
        intercept, (xslope, yslope) = self.plane.bias.detach().numpy().item(), self.plane.weight.detach().numpy()[0]
        a, b = - xslope / yslope, - intercept / yslope

        xspace = torch.tensor([X[:,0].min() * 0.75, X[:,0].max() * 1.25])
        s_1, s_2 = X_std
        m_1, m_2 = X_mean

        a = (s_2 / s_1) * a
        b = b*s_2 + m_2 - a * m_1
        print(a, b)
        plt.scatter(*X.T, c=y)

        intercept_ = - intercept + (m_1 * xslope) / s_1 + (m_2 * yslope) / s_2
        xslope_ = xslope / s_1
        yslope_ = yslope / s_2
        plotHyperplane(xspace, intercept_, xslope_, yslope_, 16)
        # plt.plot(xspace, a * xspace + b)

        xlim, ylim = get_lims(X, padding = 0.5)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.gca().set_aspect('equal')
        plt.show()


df = pd.read_csv("datasets/apples_and_oranges.csv")
model = OnePlaneModel()

X_raw = torch.tensor(df[["weight", "height"]].values, dtype=torch.float32)
y = torch.tensor(df["class"].map({"apple": 0, "orange": 1}).values, dtype=torch.float32)

X_mean = X_raw.mean(0)
X_std = X_raw.std(0)

X = (X_raw - X_mean) / X_std

model.fit(X, y.reshape(-1,1))
model.plot(X, y, X_mean, X_std)

print('LOG:\x1b[33mDEBUG\x1b[0m:', 'X:', X)
print('LOG:\x1b[33mDEBUG\x1b[0m:', 'model.plane.weight:', model.plane.weight)
print('LOG:\x1b[33mDEBUG\x1b[0m:', 'torch.column_stack([model.plane(X), y]):', torch.column_stack([model.plane(X), y]))