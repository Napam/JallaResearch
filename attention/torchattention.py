import torch
from torch import nn
import numpy as np 
from matplotlib import pyplot as plt 
import math
from debug import debug, debugs, debugt


def positional_encoding(n_tokens: int, d_model: int):
    # Get angles to be used in sin and cos
                                                             # //2 since should work on every other
    encoding = torch.arange(n_tokens).unsqueeze(1) / 10000**(2 * (torch.arange(d_model)//2) / d_model)
    
    encoding[:, 0::2] = torch.sin(encoding[:, 0::2])
    encoding[:, 1::2] = torch.cos(encoding[:, 1::2])
    return encoding.unsqueeze(0)


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
    scaled_qk = q@k.T / math.sqrt(k.shape[-1]) 
    attention = torch.softmax(scaled_qk, -1)
    return attention@v, attention


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int):
        """
        d_model: int, data dimensions (number of features)
        hidden_dim: int, dimension to map data to 
        """
        super().__init__()
        self.wq = nn.Linear(in_features=d_model, out_features=hidden_dim, bias=False)
        self.wk = nn.Linear(in_features=d_model, out_features=hidden_dim, bias=False)
        self.wv = nn.Linear(in_features=d_model, out_features=hidden_dim, bias=False)
        self.linear = nn.Linear(in_features=hidden_dim, out_features=d_model, bias=False)

    def forward(self, X: torch.Tensor):
        # Obtain query, keys and values by scaling, shearing and or rotating X with wq, wk, wv
        q = self.wq@X
        k = self.wk@X
        v = self.wv@X

        out = scaled_dot_product_attention(q, k, v)
        print(out)


if __name__ == "__main__":
    torch.set_printoptions(sci_mode=False, precision=2)

    def test_position_encoding():
        pos_encoding = positional_encoding(50, 512)
        print(pos_encoding.shape)
        plt.pcolormesh(pos_encoding[0], cmap='RdBu')
        plt.xlabel('Depth')
        plt.xlim((0, 512))
        plt.ylabel('Position')
        plt.colorbar()
        plt.show()


    def test_scaled_dot_product_attention():
        # Interpret scaled dot product attention as a continous way 
        # to search in a query-key database

        print("Scenario 1")
        # Scenario:
        # You have a query engine with four query (4x3 matrix)
        # You "search" with a single query (1x3 matrix)
        # You retrieve values, which is kinda like, 
        k = torch.tensor([ # (hidden_dim_k, n_tokens)
            [ 10,  0,  0],
            [  0, 10,  0],
            [  0,  0, 10],
            [  0,  0, 10],
        ], dtype=torch.float32)
        v = torch.tensor([ # (hidden_dim_v, idk) 
            [    1, 0],
            [   10, 0],
            [  100, 5],
            [ 1000, 6],
        ], dtype=torch.float32)
        q = torch.tensor([
            [ 0, 10, 0],
        ], dtype=torch.float32)

        out, attention = scaled_dot_product_attention(q, k, v)
        debug(attention)
        debug(out)

        print("\nScenario 2")
        # Scenerio:
        # You search with 3 queries in the database (3x3 matrix)
        q = torch.tensor([
            [  0,  0, 10],
            [  0, 10,  0],
            [ 10, 10,  0],
        ], dtype=torch.float32)
        out, attention = scaled_dot_product_attention(q, k, v)
        debug(attention)
        debug(out)


    def test_MultiHeadAttention():
        mha = MultiHeadAttention(8,8)
        assert mha.wq.bias is None
        assert mha.wk.bias is None
        assert mha.wv.bias is None

        wq = nn.Parameter(torch.eye(8))
        wk = nn.Parameter(torch.eye(8))
        wv = nn.Parameter(torch.eye(8))

        
        
    # test_scaled_dot_product_attention()
    test_MultiHeadAttention()