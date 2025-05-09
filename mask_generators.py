import torch
from torch import Tensor, device
from typing import Dict

class BaseMaskGenerator:
    def __init__(self, seq_len: int, device: device):
        self.seq_len = seq_len
        self.device = device
        self.mask = None
        self.local_count = 0

    def get_mask(self) -> Tensor:
        raise NotImplementedError("Override this method.")

    def get_stats(self) -> Dict[str, float]:
        total = self.mask.sum().item() if self.mask is not None else 0
        return {
            "total_connections": total,
            "local_connections": self.local_count,
            "shortcut_connections": total - self.local_count
        }

    def connect_cls_token(self, force: bool = False) -> None:
        """
        Ensures the CLS token (index 0) is connected to all other tokens.
        If `force=True`, overwrite existing connections; otherwise only add if missing.
        """
        if self.mask is None:
            raise ValueError("Mask must be initialized before calling connect_cls_token.")
        cls_connected_out = self.mask[0].sum().item()
        cls_connected_in = self.mask[:, 0].sum().item()
        if force or cls_connected_out < self.seq_len:
            self.mask[0, :] = True  # CLS attends to all
        if force or cls_connected_in < self.seq_len:
            self.mask[:, 0] = True  # All attend to CLS

class HubSpokeMask(BaseMaskGenerator):
    def __init__(self, seq_len: int, device: device, num_hubs: int = 8, layer_idx: int = None):
        super().__init__(seq_len, device)
        self.num_hubs = num_hubs
        self.layer_idx = layer_idx

    def get_mask(self) -> Tensor:
        mask = torch.zeros(self.seq_len, self.seq_len, dtype=torch.bool, device=self.device)
        hubs = torch.arange(self.num_hubs, device=self.device)
        if self.layer_idx is not None:
            hubs = (hubs + self.layer_idx) % self.seq_len  # circular shift for variety
        mask[:, hubs] = True
        mask[hubs, :] = True
        self.mask = mask
        return mask

class PreferentialAttachmentMask(BaseMaskGenerator):
    def __init__(self, seq_len: int, device: device, num_hubs: int = 8, layer_idx: int = None):
        super().__init__(seq_len, device)
        self.num_hubs = num_hubs
        self.layer_idx = layer_idx

    def get_mask(self) -> Tensor:
        probs = torch.linspace(1, 2, steps=self.seq_len, device=self.device)
        probs /= probs.sum()
        hubs = torch.multinomial(probs, self.num_hubs, replacement=False)
        mask = torch.zeros(self.seq_len, self.seq_len, dtype=torch.bool, device=self.device)
        mask[:, hubs] = True
        mask[hubs, :] = True
        self.mask = mask
        return mask

class SmallWorldMask(BaseMaskGenerator):
    def __init__(self, seq_len: int, device: device, num_hubs: int = 8, layer_idx: int = None):
        super().__init__(seq_len, device)
        self.num_hubs = num_hubs
        self.layer_idx = layer_idx

    def get_mask(self, shortcut_prob: float = 0.1) -> Tensor:
        mask = torch.zeros(self.seq_len, self.seq_len, dtype=torch.bool, device=self.device)
        for i in range(self.seq_len):
            start = max(0, i - self.num_hubs // 2)
            end = min(self.seq_len, i + self.num_hubs // 2)
            mask[i, start:end] = True
        self.local_count = mask.sum().item()
        num_shortcuts = int(self.seq_len * shortcut_prob)
        for _ in range(num_shortcuts):
            i, j = torch.randint(0, self.seq_len, (2,), device=self.device)
            mask[i, j] = True
        self.mask = mask
        return mask

class InfoFlowHubSpoke(BaseMaskGenerator):
    def __init__(self, seq_len: int, device: device, importance_scores: Tensor, num_hubs: int = 8, layer_idx: int = None):
        super().__init__(seq_len, device)
        self.importance_scores = importance_scores
        self.num_hubs = num_hubs
        self.layer_idx = layer_idx

    def get_mask(self) -> Tensor:
        topk = torch.topk(self.importance_scores, self.num_hubs)
        hubs = topk.indices
        mask = torch.zeros(self.seq_len, self.seq_len, dtype=torch.bool, device=self.device)
        mask[:, hubs] = True
        mask[hubs, :] = True
        self.mask = mask
        return mask

class DynamicPreferentialAttachmentMask(BaseMaskGenerator):
    def __init__(self, seq_len: int, device: device, num_hubs: int = 8, layer_idx: int = None):
        super().__init__(seq_len, device)
        self.num_hubs = num_hubs
        self.layer_idx = layer_idx

    def get_mask(self) -> Tensor:
        probs = torch.linspace(1, 2, steps=self.seq_len, device=self.device)
        probs /= probs.sum()
        g = torch.Generator(device=self.device).manual_seed(self.layer_idx) if self.layer_idx is not None else None
        hubs = torch.multinomial(probs, self.num_hubs, replacement=False, generator=g)
        mask = torch.zeros(self.seq_len, self.seq_len, dtype=torch.bool, device=self.device)
        mask[:, hubs] = True
        mask[hubs, :] = True
        self.mask = mask
        return mask

class LocalGraphMask(BaseMaskGenerator):
    def __init__(self, seq_len: int, device: device, grid_size: tuple = (14, 14), layer_idx: int = None):
        super().__init__(seq_len, device)
        assert grid_size is not None, "grid_size must be specified for 2D masking"
        self.height, self.width = grid_size
        assert self.height * self.width == seq_len, "grid_size must match seq_len"
        self.layer_idx = layer_idx

    def get_mask(self) -> Tensor:
        mask = torch.zeros((self.seq_len, self.seq_len), dtype=torch.bool, device=self.device)
        def get_index(x: int, y: int) -> int:
            return x * self.width + y
        for x in range(self.height):
            for y in range(self.width):
                i = get_index(x, y)
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < self.height and 0 <= ny < self.width:
                            j = get_index(nx, ny)
                            mask[i, j] = True
        self.mask = mask
        self.local_count = mask.sum().item()
        return mask

class Debug(BaseMaskGenerator):
    def __init__(self, seq_len: int, device: device, importance_scores: Tensor, num_hubs: int = 8, layer_idx: int = None):
        super().__init__(seq_len, device)
        self.importance_scores = importance_scores
        self.num_hubs = num_hubs
        self.layer_idx = layer_idx

    def get_mask(self) -> Tensor:
        print(self.seq_len)
        return torch.zeros((self.seq_len, self.seq_len), dtype=torch.bool, device=self.device)
