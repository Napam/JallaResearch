import torch

def get_lims(X: torch.Tensor, padding: float = 0.25):
    x_min, x_max = X[:,0].min(), X[:,0].max()
    y_min, y_max = X[:,1].min(), X[:,1].max()
    x_std, y_std = X[:,0].std(), X[:,1].std()
    return (x_min - x_std * padding, x_max + x_std * padding), (y_min - y_std * padding, y_max + y_std * padding)