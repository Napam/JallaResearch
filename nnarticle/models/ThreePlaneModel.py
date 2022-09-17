import torch
import pandas as pd
from torch import nn
from torch import optim
from matplotlib import pyplot as plt
from torch.nn.functional import one_hot
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..'))

from utils import get_lims, normalize_data, plot_hyperplane, unnormalize_plane


def mse(y_: torch.Tensor, y: torch.Tensor):
    assert y_.shape == y.shape
    return ((y_ - y) ** 2).mean()


def accuracy(y_: torch.Tensor, y: torch.Tensor):
    y_, y = y_.detach(), y.detach()
    y_ = y_.argmax(1)
    y = y.argmax(1)
    return (y_ == y).sum() / len(y)


class ThreePlaneModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.planes = nn.Linear(in_features=2, out_features=3)

    def forward(self, X: torch.Tensor):
        return torch.sigmoid(self.planes(X))

    def fit(self, X: torch.Tensor, y: torch.Tensor):
        optimizer = optim.Adam(self.parameters(), lr=1e-2, weight_decay=0.25)
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
        biases, planes = self.planes.bias.detach(), self.planes.weight.detach().numpy()
        intercept1, (xslope1, yslope1) = biases[0].item(), planes[0]
        intercept2, (xslope2, yslope2) = biases[1].item(), planes[1]
        intercept3, (xslope3, yslope3) = biases[2].item(), planes[2]
        xspace = torch.tensor([X[:, 0].min() * 0.75, X[:, 0].max() * 1.25])

        print('LOG:\x1b[33mDEBUG\x1b[0m:', 'intercept1:', intercept1)
        print('LOG:\x1b[33mDEBUG\x1b[0m:', 'xslope1:', xslope1)
        print('LOG:\x1b[33mDEBUG\x1b[0m:', 'yslope1:', yslope1)

        print('LOG:\x1b[33mDEBUG\x1b[0m:', 'intercept2:', intercept2)
        print('LOG:\x1b[33mDEBUG\x1b[0m:', 'xslope2:', xslope2)
        print('LOG:\x1b[33mDEBUG\x1b[0m:', 'yslope2:', yslope2)

        print('LOG:\x1b[33mDEBUG\x1b[0m:', 'intercept3:', intercept3)
        print('LOG:\x1b[33mDEBUG\x1b[0m:', 'xslope3:', xslope3)
        print('LOG:\x1b[33mDEBUG\x1b[0m:', 'yslope3:', yslope3)

        plt.scatter(*X.T, c=y)
        intercept_1, xslope_1, yslope_1 = unnormalize_plane(X_mean, X_std, intercept1, xslope1, yslope1)
        intercept_2, xslope_2, yslope_2 = unnormalize_plane(X_mean, X_std, intercept2, xslope2, yslope2)
        intercept_3, xslope_3, yslope_3 = unnormalize_plane(X_mean, X_std, intercept3, xslope3, yslope3)
        plot_hyperplane(xspace, intercept_1, xslope_1, yslope_1, 16, c='red', quiver_kwargs={'scale': 0.05, 'units': 'dots', 'width': 2})
        plot_hyperplane(xspace, intercept_2, xslope_2, yslope_2, 16, c='green', quiver_kwargs={'scale': 0.05, 'units': 'dots', 'width': 2})
        plot_hyperplane(xspace, intercept_3, xslope_3, yslope_3, 16, c='blue', quiver_kwargs={'scale': 0.05, 'units': 'dots', 'width': 2})

        xlim, ylim = get_lims(X, padding=0.5)
        plt.xlim(*xlim)
        plt.ylim(*ylim)
        plt.gca().set_aspect('equal')
        plt.show()


df = pd.read_csv("../datasets/apples_oranges_pears.csv")
model = ThreePlaneModel()

X_raw = torch.tensor(df[["weight", "height"]].values, dtype=torch.float32)
y_raw = torch.tensor(df["class"].map({"apple": 0, "orange": 1, "pear": 2}).values, dtype=torch.long)

X, X_mean, X_std = normalize_data(X_raw)
y = one_hot(y_raw)

model.fit(X, y)
model.plot(X, y, X_mean, X_std)
print('LOG:\x1b[33mDEBUG\x1b[0m:', 'X_mean:', X_mean)
print('LOG:\x1b[33mDEBUG\x1b[0m:', 'X_std:', X_std)
