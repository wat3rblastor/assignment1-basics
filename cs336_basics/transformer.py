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
      torch.zeros((out_features, in_features),
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
      torch.zeros(
        (num_embeddings, embedding_dim),
        device=device,
        dtype=dtype
      )
    )
    
    torch.nn.init.trunc_normal_(self.embedding_matrix, 0, 1, -3, 3) 
    
  def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
    return self.embedding_matrix[token_ids]