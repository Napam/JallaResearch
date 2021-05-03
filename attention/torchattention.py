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
    def __init__(self, d_model: int, hidden_dim_qk: int, hidden_dim_v: int, n_heads: int):
        """
        d_model: int, data dimensions (number of features)
        hidden_dim: int, dimension to map data to 
        """
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.wq = nn.Linear(in_features=d_model, out_features=hidden_dim_qk, bias=False)
        self.wk = nn.Linear(in_features=d_model, out_features=hidden_dim_qk, bias=False)
        self.wv = nn.Linear(in_features=d_model, out_features=hidden_dim_v, bias=False)
        self.linear = nn.Linear(in_features=hidden_dim_qk, out_features=d_model, bias=False)

    def split_heads(self, X: torch.tensor):
        return X.view(-1, self.n_heads, self.d_head).permute(1,0,2)

    def forward(self, Xq: torch.Tensor, Xk: torch.Tensor, Xv: torch.Tensor):
        # Obtain query, keys and values by scaling, shearing and or rotating X with wq, wk, wv
        q = self.wq(Xq)
        k = self.wk(Xk)
        v = self.wv(Xv)

        q = self.split_heads(q)
        k = self.split_heads(k)
        v = self.split_heads(v)

        # debug(q)
        # debug(k)
        # debug(k.permute(1,2,0).T)
        # debug(q@k.permute(1,2,0).T)
        out, attention = scaled_dot_product_attention(q, k.permute(1,2,0), v)
        return out.permute(1,0,2).reshape(-1,self.d_model)


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

        print(
            "\x1b[1m\x1b[3mScenario 1:\x1b[0m\n"
            "You have a query engine with four queries (4x3 matrix)\n"
            "You \"search\" with a single query (1x3 matrix)\n"
            "You retrieve values"
        )
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
            [    1,   0],
            [   10,   0],
            [  100,   5],
            [ 1000,   6],
        ], dtype=torch.float32)
        q = torch.tensor([
            [ 0, 10, 0],
        ], dtype=torch.float32)

        out, attention = scaled_dot_product_attention(q, k, v)
        debug(attention)
        debug(out)

        print(
            "\x1b[1m\x1b[3mScenario 2:\x1b[0m\n"
            "You search with 3 queries in the data base (query is 3x3 matrox)"
        )
        q = torch.tensor([
            [  0,  0, 10],
            [  0, 10,  0],
            [ 10, 10,  0],
        ], dtype=torch.float32)
        out, attention = scaled_dot_product_attention(q, k, v)
        debug(attention)
        debug(out)

    @torch.no_grad()
    def test_MultiHeadAttention():
        def singlehead():
            dim = 3
            mha = MultiHeadAttention(d_model=3, hidden_dim_qk=3, hidden_dim_v=2, n_heads=1)
            assert mha.wq.bias is None
            assert mha.wk.bias is None
            assert mha.wv.bias is None

            mha.wq.weight = nn.Parameter(torch.eye(dim))
            mha.wk.weight = nn.Parameter(torch.eye(dim))
            mha.wv.weight = nn.Parameter(torch.eye(dim))

            Xq = torch.tensor([
                [  0,  0, 10],
                [  0, 10,  0],
                [ 10, 10,  0],
            ], dtype=torch.float32)

            Xk = torch.tensor([ 
                [ 10,  0,  0],
                [  0, 10,  0],
                [  0,  0, 10],
                [  0,  0, 10],
            ], dtype=torch.float32)

            Xv = torch.tensor([
                [    1,  0, 0],
                [   10,  0, 0],
                [  100,  5, 0],
                [ 1000,  6, 0], 
            ], dtype=torch.float32)

            print(
                "\x1b[1m\x1b[3mNo multi head attention (regular matmul)\x1b[0m"
            )
            out = mha(Xq, Xk, Xv)
            debug(out)

        # singlehead()

        def multihead():
            print(
                "\x1b[1m\x1b[3m2-head multi head attention \x1b[0m"
            )
            dim = 4
            mha = MultiHeadAttention(d_model=dim, hidden_dim_qk=dim, hidden_dim_v=2, n_heads=2)

            Xq = torch.tensor([
                [  0, 10, 0, 0],
                [  0, 10, 0, 0],
                [ 10, 10, 0, 0],
                [  0,  0, 0,10],
            ], dtype=torch.float32)

            Xk = torch.tensor([ 
                [ 10,  0,  0,  0],
                [  0, 10,  0,  0],
                [  0,  0, 10,  0],
                [  0,  0, 10,  0],
                [  0,  0,  0, 10],
            ], dtype=torch.float32)

            Xv = torch.tensor([
                [    1,   0,   0,   0],
                [   10,   0,   0,   0],
                [  100,   5,   0,   0],
                [ 1000,   6,   0,   0],
                [  200,   6,   0,   0],
            ], dtype=torch.float32)

            mha.wq.weight = nn.Parameter(torch.eye(dim))
            mha.wk.weight = nn.Parameter(torch.eye(dim))
            mha.wv.weight = nn.Parameter(torch.eye(Xv.shape[1]))

            out = mha(Xq, Xk, Xv)
            debug(out)

        multihead()
        
    # test_scaled_dot_product_attention()
    test_MultiHeadAttention()