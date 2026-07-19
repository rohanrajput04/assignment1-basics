import torch 
import torch.nn as nn
from einops import einsum, rearrange
from torch import Tensor
from jaxtyping import Bool, Float, Int
import math
import numpy as np
import os 
import typing

import numpy.typing as npt

from collections.abc import Callable, Iterable
from typing import Optional

def silu(x):
    return x * torch.sigmoid(x)

def softmax(x, dim):
    shifted = x - x.max(dim, keepdim=True).values
    e = shifted.exp()
    return e / e.sum(dim, keepdim=True)

def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... keys d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None ) -> Float[Tensor, "... queries d_v"]:

    scores = einsum(Q, K, "... q d, ... k d -> ... q k") / math.sqrt(Q.shape[-1])

    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    scaled_softmax = softmax(scores, dim=-1)
    
    weighted_values = einsum(scaled_softmax, V, "... q k, ... k v -> ... q v")

    return weighted_values



class Linear(nn.Module):
    def __init__(self, in_features: int, out_features:int, 
    device: torch.device | None = None, dtype: torch.dtype | None = None):
        """
        Constructs a linear class 

        args:
        in_featurs: int final dimension of input
        out_features: int final dimension of output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        
        # create variance
        variance = 2 / (in_features + out_features)
        std = (variance)**0.5

        # weight tensor 
        w_tensor = torch.empty(out_features, in_features,device=device, dtype=dtype)

        #initialize
        nn.init.trunc_normal_(w_tensor, mean=0.0, std=std, a=-3*std, b=3*std)

        self.weight = nn.Parameter(w_tensor)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies linear transformation to the input"""
        # implement Y = WX linear transformation
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    """Embedding layer for the neural network

    Args:
        nn (_type_): _description_
    """

    def __init__(self, num_embedding: int, embedding_dim: int, 
    device: torch.device | None = None, dtype: torch.dtype | None = None):
        """Contructs the embedding module

        Args:
            num_embedding (int): Size of vocabulary
            embedding_dim (int): Dimension of the embedding vectos, d_model
            device (torch.device | None, optional): Device to store parameters. Defaults to None.
            dtype (torch.dtype | None = Nobe, optional): Data type of the parameters
        """
        super().__init__()
        # weight tensor
        w_tensor = torch.empty(num_embedding, embedding_dim,device=device, dtype=dtype)
        # initalize 
        nn.init.trunc_normal_(w_tensor, mean=0, std=1, a=-3, b=3)

        self.weight = nn.Parameter(w_tensor)



    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """Apply linear transformation to the input

        Args:
            x (torch.Tensor): input 

        Returns:
            torch.Tensor: output
        """
        return self.weight[x]

class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5, device: torch.device | None = None, dtype: torch.dtype | None = None):
        super().__init__()

        self.eps = eps 
        w_tensor = torch.ones(d_model, device=device, dtype=dtype)
        self.weight = nn.Parameter(w_tensor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        in_type = x.dtype
        x = x.to(torch.float32)
        # RMSNorm
        mean_sq = x.pow(2).mean(dim=-1, keepdim=True)
        rms = (mean_sq + self.eps).sqrt()

        result = (x / rms) * self.weight


        return result.to(in_type)

class SwiGLU(nn.Module):

    def __init__(self, d_model, d_ff, device=None, dtype=None):
        
        super().__init__()

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device,dtype=dtype)
    
    def forward(self, x):
        
        return self.w2(silu(self.w1(x)) * self.w3(x))

class SiLUFFN(nn.Module):

    def __init__(self, d_model, d_ff, device=None, dtype=None):

        super().__init__()

        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
    
    def forward(self,x):
        return self.w2(silu(self.w1(x)))

class RoPE(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq: int, device=None):
        super().__init__()

        # build inverse frequecy
        theta_arrange = torch.arange(0, d_k, 2)
        inv_freq = 1.0 / (theta ** (theta_arrange / d_k))

        # Angles = outer product of positions and inv_freq 
        positions = torch.arange(max_seq) # shape (max_seq_len, )
        angles = torch.outer(positions, inv_freq) # shape (max_seq_len, d_k/2)

        # register tables 
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        
        # index token position
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]

        # split the x in even odd
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # apply rotation 
        out_even = x_even * cos - x_odd * sin 
        out_odd = x_even * sin + x_odd * cos

        return torch.stack([out_even, out_odd], dim=-1).flatten(-2)

class MultiHeadSelfAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, max_seq_len: int = None, theta: float = None, 
    use_rope : bool=False, device: torch.device = None):
        super().__init__()

        # store d_k
        d_k = d_model // num_heads

        assert d_model % num_heads == 0

        self.d_k = d_k
        self.d_model = d_model
        self.num_heads = num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

        # use rope 
        self.rope = RoPE(theta=theta, d_k=self.d_k, max_seq=max_seq_len, device=device) if use_rope else None 

    
    def forward(self, x: torch.Tensor, token_position=None):

        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        # Q, K , V dim rearrange to heads
        Q = rearrange(Q, "... seq (h dk) -> ... h seq dk", h=self.num_heads)
        K = rearrange(K, "... seq (h dk) -> ... h seq dk", h=self.num_heads)
        V = rearrange(V, "... seq (h dk) -> ... h seq dk", h=self.num_heads)

        # causal mask
        seq = x.shape[-2]

        if token_position is None:
            token_position = torch.arange(seq, device=x.device)

        # if rope applied 
        if self.rope is not None:
            Q = self.rope(Q, token_position)
            K = self.rope(K, token_position)
        

        causal_mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))

        attn = scaled_dot_product_attention(Q, K, V, mask=causal_mask)

        attn = rearrange(attn, "... h seq dk -> ... seq (h dk)")

        return self.output_proj(attn)

class TransformerBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float, 
    device: torch.device = None, use_rope : bool = True, ffn_type = "swiglu", norm_position="pre",use_rmsnorm = True):

        super().__init__()

        self.norm_position = norm_position
        # first layer norm 
        if use_rmsnorm:
            self.ln1 = RMSNorm(d_model, device=device)
        else:
            self.ln1 = nn.Identity(d_model, device=device)

        # attention 
        self.attn = MultiHeadSelfAttention(d_model, num_heads, max_seq_len, theta, use_rope=use_rope,device=device)

        # second layer norm
        if use_rmsnorm:
            self.ln2 = RMSNorm(d_model,device=device)
        else:
            self.ln2 = nn.Identity(d_model, device=device)

        # FFN 
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff)
        else:
            self.ffn = SiLUFFN(d_model, d_ff)

    def forward(self, x, token_positions=None):
        if self.norm_position == "pre":
            y = x + self.attn(self.ln1(x), token_positions)
            return y + self.ffn(self.ln2(y))
        else:
            y = self.ln1(x + self.attn(x, token_positions))
            return self.ln2(y + self.ffn(y))


class TransformerLM(nn.Module):

    def __init__(self, vocab_size, context_length, d_model, num_layers, num_heads, d_ff, rope_theta, 
    use_rope=True, ffn_type="swiglu", norm_position = "pre", use_rmsnorm = True):
        super().__init__()

        self.use_rmsnorm = use_rmsnorm

        # token embedding 
        self.token_embeddings = Embedding(vocab_size, d_model)

        # transformer block stack 
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, use_rope=use_rope, ffn_type=ffn_type, norm_position=norm_position, use_rmsnorm=use_rmsnorm)
            for _ in range(num_layers)
        ])

        # layer norm on final layer
        self.ln_final = RMSNorm(d_model)

        # linear to vocab 
        self.lm_head = Linear(d_model, vocab_size)
    
    def forward(self, in_indices):
        x = self.token_embeddings(in_indices)

        seq = in_indices.shape[-1]

        token_positions = torch.arange(seq, device=in_indices.device)

        # loop blocks
        for block in self.layers:
            x = block(x, token_positions)
        
        # final norm 
        if self.use_rmsnorm:
            x = self.ln_final(x)
        else:
            x = nn.Identity()(x)

        # logits 
        logits = self.lm_head(x)

        return logits


def cross_entropy(logits: Float[Tensor, " batch_size vocab_size"], target: Int[Tensor, " batch_size"]):

    m = logits.max(dim=-1, keepdim=True).values
    logsumexp = (logits - m).exp().sum(dim=-1)
    logsumexp = logsumexp.log() + m.squeeze(-1)
    o_target = torch.gather(logits, -1, target.unsqueeze(-1)).squeeze(-1)

    per_example_loss = logsumexp - o_target

    return per_example_loss.mean()
    


class SGD(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr" : lr}
        super().__init__(params, defaults)
    
    def step(self, closure: Optional[Callable] | None = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data 
                p.data -= lr / math.sqrt(t+1) * grad
                state["t"] = t + 1
        return loss


class AdamW(torch.optim.Optimizer):

    def __init__(self, params, lr, betas = (0.9, 0.999), eps=1e-8, weight_decay=0.01):
        
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr" : lr, "betas" : betas, "eps": eps, "weight_decay" : weight_decay}
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr, betas, eps, weight_decay = group["lr"], group["betas"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None : continue
                grad = p.grad.data
                state = self.state[p]

                # AdamW 
                # initalized
                if not state:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)
                m, v = state["m"], state["v"]
                t = state["t"] + 1

                # weight decay 
                p.data -= lr * weight_decay * p.data 

                # update momentum
                m = betas[0] * m + (1-betas[0])*grad
                v = betas[1] * v + (1 - betas[1])*grad**2

                # bais correction 
                alpha_t = lr * math.sqrt(1 - betas[1]**t) / (1 - betas[0]**t)

                # param updates
                p.data -= alpha_t * m / (v.sqrt() + eps)
               

                # write back 
                state["t"], state["m"], state["v"] = t, m, v
        return loss

def get_lr_cosine_schedule(t, alpha_max, alpha_min, T_w, T_c):
    if t < T_w:
        alpha_t = (t / T_w) * alpha_max
    elif T_w <= t <= T_c:
        alpha_t = alpha_min + 0.5 * (1 + math.cos((t - T_w)/ (T_c - T_w) * math.pi)) * (alpha_max - alpha_min)
    else:
        alpha_t = alpha_min
    return alpha_t

def gradient_cliping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    sum_sq = 0
    parameters = list(parameters)
    for p in parameters:
        if p.grad is not None:
            sum_sq += p.grad.pow(2).sum()
        
    total_norm = torch.sqrt(sum_sq)

    if total_norm > max_l2_norm:
        coef = max_l2_norm / (total_norm + 1e-6)
        for p in parameters:
            if p.grad is not None:
                p.grad *= coef
    

def get_batch(dataset: npt.NDArray, batch_size: int, context_length: int, device: str):
    n = len(dataset)
    m = context_length
    start_idx = np.random.randint(0 , n - m, batch_size)

    start_idx = start_idx.reshape(batch_size, 1)
    row = np.arange(m).reshape(1,m)
    input_idx_grid = start_idx + row 
    target_idx_grid = input_idx_grid + 1

    input_batch = torch.from_numpy(dataset[input_idx_grid]).long().to(device)
    target_batch = torch.from_numpy(dataset[target_idx_grid]).long().to(device)

    return input_batch, target_batch

def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, iteration: int, out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes]  ):
    state_dict = {"model" : model.state_dict(), "optimizer" : optimizer.state_dict(), "iteration": iteration}
    torch.save(state_dict, out)

def load_checkpoint(src : str | os.PathLike | typing.BinaryIO | typing.IO[bytes], model : torch.nn.Module, optimizer : torch.optim.Optimizer ):
    state_dict = torch.load(src)
    model.load_state_dict(state_dict["model"])
    optimizer.load_state_dict(state_dict["optimizer"])
    return state_dict["iteration"]