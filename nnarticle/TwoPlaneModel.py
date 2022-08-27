from typing import Optional
import torch
import pandas as pd
from torch import nn
from torch import optim
from matplotlib import pyplot as plt
from torch.nn.functional import binary_cross_entropy, one_hot
from utils import get_lims, normalize_data, plot_hyperplane, unnormalize_plane
import sys


def mse(y_: torch.Tensor, y: torch.Tensor):
    assert y_.shape == y.shape
    return ((y_ - y) ** 2).mean()


def accuracy(y_: torch.Tensor, y: torch.Tensor):
    y_, y = y_.detach(), y.detach()
    y_ = y_.argmax(1)
    y = y.argmax(1)
    return (y_ == y).sum() / len(y)


class TwoPlaneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.plane1 = nn.Linear(in_features=2, out_features=1)
        self.plane2 = nn.Linear(in_features=2, out_features=1)

    def forward(self, X: torch.Tensor):
        y_ = torch.sigmoid(torch.cat([self.plane1(X), self.plane2(X)], dim=1))
        # Add column as a pseudo-node to represent missing plane for class 1 (which should be orange)
        return torch.column_stack([y_[:, 0], (y_ < 0.5).all(1) * (1 - y_.sum(1) / 2), y_[:, 1]])

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        optimizer = optim.Adam(self.parameters(), lr=1e-2)
        criterion = mse

        losses = []
        for i in range(1000):
            y_ = self.forward(X)
            loss = criterion(y_, y)
            print(f"Loss: {loss.item():<25} Accuracy: {accuracy(y_, y).item()}")
            if losses and torch.allclose(losses[-1], loss, atol=1e-5):
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
        intercept1, (xslope1, yslope1) = self.plane1.bias.detach().numpy().item(), self.plane1.weight.detach().numpy()[0]
        intercept2, (xslope2, yslope2) = self.plane2.bias.detach().numpy().item(), self.plane2.weight.detach().numpy()[0]
        xspace = torch.tensor([X[:, 0].min() * 0.75, X[:, 0].max() * 1.25])

        print('LOG:\x1b[33mDEBUG\x1b[0m:', 'intercept1:', intercept1)
        print('LOG:\x1b[33mDEBUG\x1b[0m:', 'xslope1:', xslope1)
        print('LOG:\x1b[33mDEBUG\x1b[0m:', 'yslope1:', yslope1)

        print('LOG:\x1b[33mDEBUG\x1b[0m:', 'intercept2:', intercept2)
        print('LOG:\x1b[33mDEBUG\x1b[0m:', 'xslope2:', xslope2)
        print('LOG:\x1b[33mDEBUG\x1b[0m:', 'yslope2:', yslope2)

        plt.scatter(*X.T, c=y)
        intercept_1, xslope_1, yslope_1 = unnormalize_plane(X_mean, X_std, intercept1, xslope1, yslope1)
        intercept_2, xslope_2, yslope_2 = unnormalize_plane(X_mean, X_std, intercept2, xslope2, yslope2)
        plot_hyperplane(xspace, intercept_1, xslope_1, yslope_1, 16, c='red', quiver_kwargs={'scale': 0.05, 'units': 'dots', 'width': 2})
        plot_hyperplane(xspace, intercept_2, xslope_2, yslope_2, 16, c='blue', quiver_kwargs={'scale': 0.05, 'units': 'dots', 'width': 2})

        xlim, ylim = get_lims(X, padding=0.5)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.gca().set_aspect('equal')
        plt.show()


df = pd.read_csv("datasets/apples_oranges_pears.csv")
model = TwoPlaneModel()

X_raw = torch.tensor(df[["weight", "height"]].values, dtype=torch.float32)
y_raw = torch.tensor(df["class"].map({"apple": 0, "orange": 1, "pear": 2}).values, dtype=torch.long)

X, X_mean, X_std = normalize_data(X_raw)
y = one_hot(y_raw)

model.fit(X, y)
model.plot(X, y, X_mean, X_std)
print('LOG:\x1b[33mDEBUG\x1b[0m:', 'X_mean:', X_mean)
print('LOG:\x1b[33mDEBUG\x1b[0m:', 'X_std:', X_std)
