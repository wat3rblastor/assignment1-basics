import torch
import math

class Linear(torch.nn.Module):
  def __init__(self, 
               in_features: int, 
               out_features: int, 
               device: torch.device | None = None,
               dtype: torch.dtype | None = None):
    super().__init__()
    
    self.w = torch.nn.parameter.Parameter(
      torch.empty((out_features, in_features),
                  device=device,
                  dtype=dtype
      )
    )
    
    variance = 2 / (in_features + out_features)
    stddev = math.sqrt(variance)
    torch.nn.init.trunc_normal_(self.w, 0, stddev, -3 * stddev, 3 * stddev)
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x @ self.w.T


class Embedding(torch.nn.Module):
  def __init__(self,
               num_embeddings: int,
               embedding_dim: int,
               device: torch.device | None = None,
               dtype: torch.dtype | None = None):
    super().__init__()
    
    self.embedding_matrix = torch.nn.parameter.Parameter(
      torch.empty(
        (num_embeddings, embedding_dim),
        device=device,
        dtype=dtype
      )
    )
    
    torch.nn.init.trunc_normal_(self.embedding_matrix, 0, 1, -3, 3) 
    
  def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
    return self.embedding_matrix[token_ids]
  
  
class RMSNorm(torch.nn.Module):
  def __init__(self,
               d_model: int,
               eps: float = 1e-5,
               device: torch.device | None = None,
               dtype: torch.dtype | None = None):
    super().__init__()
    
    self.g = torch.nn.parameter.Parameter(
      torch.ones(d_model, device=device, dtype=dtype)
    )
    
    self.d_model = d_model
    self.eps = eps
  
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    in_dtype = x.dtype
    x = x.to(torch.float32)
    
    rms = torch.sqrt(
      x.pow(2).mean(dim=-1, keepdim=True) + self.eps
    )
    
    result = (1 / rms) * x * self.g

    return result.to(in_dtype)
  
  
class SiLU(torch.nn.Module):
  def __init__(self):
    super().__init__()
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)
  
  
class FeedForward(torch.nn.Module):
  def __init__(self,
               d_model: int,
               d_ff: int,
               device: torch.device | None = None,
               dtype: torch.dtype | None = None):
    super().__init__()
    
    self.d_model = d_model
    self.d_ff = d_ff
    self.silu = SiLU()

    self.w1 = torch.nn.parameter.Parameter(
      torch.empty(
        (self.d_ff, self.d_model),
        device=device,
        dtype=dtype
      )
    )
    
    self.w2 = torch.nn.parameter.Parameter(
      torch.empty(
        (self.d_model, self.d_ff),
        device=device,
        dtype=dtype
      )
    )
    
    self.w3 = torch.nn.parameter.Parameter(
      torch.empty(
        (self.d_ff, self.d_model),
        device=device,
        dtype=dtype
      )
    )
    
    variance = 2 / (self.d_model + self.d_ff)
    stddev = math.sqrt(variance)
    
    torch.nn.init.trunc_normal_(self.w1, 0, stddev, -3 * stddev, 3 * stddev)
    torch.nn.init.trunc_normal_(self.w2, 0, stddev, -3 * stddev, 3 * stddev)
    torch.nn.init.trunc_normal_(self.w3, 0, stddev, -3 * stddev, 3 * stddev)
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    w1_x = x @ self.w1.T
    silu_x = self.silu(w1_x)
    
    w3_x = x @ self.w3.T
    
    return (silu_x * w3_x) @ self.w2.T
  
  
class RotaryPositionalEmbedding(torch.nn.Module):
  def __init__(self,
                theta: float,
                d_k: int,
                max_seq_len: int,
                device: torch.device | None = None):
    super().__init__()
    
    i = torch.arange(max_seq_len, device=device).unsqueeze(1)
    k = torch.arange(d_k // 2, device=device).unsqueeze(0)
    
    angle = i / (theta ** (2 * k / d_k))
    
    sin = torch.sin(angle)
    cos = torch.cos(angle)
    
    self.sin_buffer: torch.Tensor
    self.cos_buffer: torch.Tensor
    
    self.register_buffer("sin_buffer", sin, persistent=False)
    self.register_buffer("cos_buffer", cos, persistent=False)
  
  def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
    sin = self.sin_buffer[token_positions]
    cos = self.cos_buffer[token_positions] # (..., seq_len, d_k // 2)
    
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    
    x_rot_even = cos * x_even - sin * x_odd
    x_rot_odd = sin * x_even + cos * x_odd
    
    x_out = torch.empty_like(x)
    
    x_out[..., ::2] = x_rot_even
    x_out[..., 1::2] = x_rot_odd
  
    return x_out
  
  
def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
  x = x - torch.amax(x, dim=dim, keepdim=True)
  x_exp = torch.exp(x)

  return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(
  Q: torch.Tensor,
  K: torch.Tensor,
  V: torch.Tensor,
  mask: torch.Tensor | None = None
) -> torch.Tensor:
  d_k = Q.shape[-1]
  
  pre_softmax = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)
  
  if mask is not None:
    pre_softmax = pre_softmax.masked_fill(~mask, float("-inf"))
  
  value_weights = softmax(pre_softmax, -1)
  
  return value_weights @ V