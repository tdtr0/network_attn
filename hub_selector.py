import torch
from torch import nn
from CSE487.mask_generators import BaseMaskGenerator  # For type hints

class HubSelector(nn.Module):
    def __init__(self, seq_len: int, num_hubs: int):
        super().__init__()
        self.seq_len: int = seq_len
        self.num_hubs: int = num_hubs
        self.scores: nn.Parameter = nn.Parameter(torch.randn(seq_len))

    def forward(self) -> torch.Tensor:
        # Returns indices of top-k hubs
        return torch.topk(self.scores, self.num_hubs).indices
