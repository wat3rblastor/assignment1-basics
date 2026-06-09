import torch
import math

class Linear(torch.nn.Module):
  def __init__(self, 
               in_features: int, 
               out_features: int, 
               device: torch.device | None = None,
               dtype: torch.dtype | None = None):
    super().__init__()
    
    self.w = torch.nn.parameter.Parameter(torch.zeros((out_features, in_features), dtype=dtype))
    
    variance = 2 / (in_features + out_features)
    stddev = math.sqrt(variance)
    torch.nn.init.trunc_normal_(self.w, 0, stddev, -3 * stddev, 3 * stddev)
    
    self.device = device
    self.dtype = dtype
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x @ self.w.T

  