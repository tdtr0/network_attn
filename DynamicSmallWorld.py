import torch
import math
# Try relative import first for script usage
try:
    from mask_generators import BaseMaskGenerator
except ImportError:
    from CSE487.mask_generators import BaseMaskGenerator

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import os
from PIL import Image
import matplotlib.pyplot as plt

try:
    from transformers import CLIPProcessor
except ImportError:
    CLIPProcessor = None

class SmallWorldDebug(BaseMaskGenerator):
    def __init__(self, seq_len, device, grid_shape=None, distance_type='manhattan',
                 radius=4, shortcut_prob=0.5, wrap_around=True, verbose=False):
        super().__init__(seq_len, device)
        assert distance_type in ['manhattan', 'euclidean'], "distance_type must be 'manhattan' or 'euclidean'"
        self.seq_len = seq_len
        self.device = device
        self.distance_type = distance_type
        self.radius = radius
        self.shortcut_prob = shortcut_prob
        self.wrap_around = wrap_around
        self.verbose = verbose

        if grid_shape:
            self.H, self.W = grid_shape
            self.fixed_shape = (self.H, self.W)
        else:
            self.H = int((seq_len - 1) ** 0.5)
            self.W = self.H
            assert self.H * self.W + 1 == seq_len, "seq_len must be H*W+1 for grid + CLS"
            self.fixed_shape = (self.H, self.W)

        self.coords = [(i, j) for i in range(self.H) for j in range(self.W)]

    def _rebuild_grid(self):
        if hasattr(self, 'fixed_shape'):
            self.H, self.W = self.fixed_shape
        else:
            H = int((self.seq_len - 1) ** 0.5)
            W = H
            assert H * W + 1 == self.seq_len, "seq_len must be H*W+1"
            self.fixed_shape = (H, W)
            self.H, self.W = H, W
        self.coords = [(i, j) for i in range(self.H) for j in range(self.W)]
        if self.verbose:
            print(f"[Debug] Grid rebuilt to H={self.H}, W={self.W}")

    def coord_to_idx(self, x, y):
        return x * self.W + y

    def get_neighbors(self, x, y):
        neighbors = []
        for dx in range(-self.radius, self.radius + 1):
            for dy in range(-self.radius, self.radius + 1):
                if dx == 0 and dy == 0:
                    continue
                if self.wrap_around:
                    nx = (x + dx) % self.H
                    ny = (y + dy) % self.W
                else:
                    nx = x + dx
                    ny = y + dy
                    if not (0 <= nx < self.H and 0 <= ny < self.W):
                        continue
                if self.distance_type == 'manhattan':
                    if abs(dx) + abs(dy) <= self.radius:
                        neighbors.append((nx, ny))
                else:
                    if math.sqrt(dx**2 + dy**2) <= self.radius:
                        neighbors.append((nx, ny))
        if self.verbose:
            print(f"[Debug] Neighbors for ({x},{y}): {len(neighbors)}")
        return neighbors

    def get_mask(self):
        self._rebuild_grid()
        N = self.seq_len  # H*W + 1
        mask = torch.zeros((N, N), dtype=torch.bool, device=self.device)

        # Local connections (between patch tokens, indices 1 to N-1)
        # grid_idx runs from 0 to H*W-1. These map to mask indices 1 to N-1.
        for grid_idx_src, (x, y) in enumerate(self.coords):
            mask_idx_src = grid_idx_src + 1  # Offset by 1 for CLS token
            neighbors = self.get_neighbors(x, y)
            for nx, ny in neighbors:
                grid_idx_tgt = self.coord_to_idx(nx, ny)
                mask_idx_tgt = grid_idx_tgt + 1  # Offset by 1 for CLS token
                mask[mask_idx_src, mask_idx_tgt] = True
        
        if self.verbose:
            # Count connections only within the patch part of the mask for "local"
            patch_local_cnt = mask[1:N, 1:N].sum().item()
            print(f"[Debug] Local patch-to-patch connections count: {patch_local_cnt}")

        # Shortcuts (primarily between patch tokens, indices 1 to N-1)
        # We'll add a number of shortcuts proportional to N, but target them to patches.
        num_shortcuts_to_add = int(N * self.shortcut_prob) 
        if self.verbose:
            print(f"[Debug] Attempting to add {num_shortcuts_to_add} shortcuts between patch tokens")

        shortcuts_actually_added = 0
        if N > 1: # Ensure there are patch tokens to connect
            for _ in range(num_shortcuts_to_add):
                # randint high is exclusive, so N-1 for patches 0..N-2, then +1 for mask indices 1..N-1
                # Simpler: randint(1, N) gives from 1 to N-1 inclusive for mask indices
                idx_i = torch.randint(1, N, (1,), device=self.device).item()
                idx_j = torch.randint(1, N, (1,), device=self.device).item()
                if idx_i != idx_j:
                    # Add the shortcut, potentially overwriting an existing local one
                    # or adding a new one. Could also check `if not mask[idx_i, idx_j]:` 
                    # if we only want to add to non-existing edges.
                    mask[idx_i, idx_j] = True
                    shortcuts_actually_added +=1 # Counts attempts that become connections
        
        if self.verbose:
            print(f"[Debug] Actual shortcuts added between patch tokens: {shortcuts_actually_added}")
            # patch_total_after_shortcuts = mask[1:N, 1:N].sum().item()
            # print(f"[Debug] Total patch-to-patch connections after shortcuts: {patch_total_after_shortcuts}")

        # CLS token (index 0) fully connected
        mask[0, :] = True  # CLS token attends to all tokens (including itself)
        mask[:, 0] = True  # All tokens attend to CLS token
        
        if self.verbose:
            total_cnt = mask.sum().item()
            print(f"[Debug] Total connections in final mask (incl. CLS): {total_cnt}")

        self.mask = mask
        return mask


# --- Data Loading ---
class CLIPCaptionDataset(Dataset):
    def __init__(self, image_dir="/home/thanhdo/CSE487/laion400m", captions_file="/home/thanhdo/CSE487/laion400m/captions.txt"):
        self.image_dir = image_dir
        self.data = []
        with open(captions_file, "r") as f:
            for line in f:
                if "\t" in line:
                    filename, caption = line.strip().split("\t", 1)
                    self.data.append((filename, caption))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        filename, caption = self.data[idx]
        image_path = os.path.join(self.image_dir, filename)
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
        except Exception as e:
            print(f"[Skip] Bad image at {image_path}: {e}")
            return None
        return {"image": image, "text": caption}

def collate_fn(batch):
    # filter out failed samples
    batch = [item for item in batch if item is not None]
    if not batch:
        return {"image": [], "text": []}
    images = [item["image"] for item in batch]
    texts = [item["text"] for item in batch]
    return {"image": images, "text": texts}

# --- Evaluation ---
def evaluate_clip_laion(model, processor, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch["image"]
            texts = batch["text"]
            # Ensure all images are 3-channel RGB PIL Images
            processed_images = []
            for img in images:
                if not hasattr(img, 'mode') or img.mode != 'RGB':
                    try:
                        img = img.convert('RGB')
                        print("[Warning] Converted image to RGB.")
                    except Exception as e:
                        print(f"[Error] Could not convert image to RGB: {e}")
                        from PIL import Image
                        img = Image.new("RGB", (224, 224))  # fallback dummy image
                processed_images.append(img)
            inputs = processor(text=texts, images=processed_images, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image
            preds = logits_per_image.argmax(dim=-1)
            ground_truth = torch.arange(len(images), device=device)
            correct += (preds == ground_truth).sum().item()
            total += len(images)
    acc = correct / total
    return acc

# --- Plotting/Experiment Loop Example ---
def run_smallworld_debug_experiment(clip_base, processor, dataloader, device, shortcut_probs, grid_shape, radius, distance_type, wrap_around=True, verbose=False):
    accuracies = []



    for p in shortcut_probs:
        print(f"\nEvaluating with shortcut_prob={p:.2f}…")
        import copy
        clip_masked = copy.deepcopy(clip_base)
        vision_layers = clip_masked.vision_model.encoder.layers
        try:
            from patching import patch_model_attention_with_mask
        except ImportError:
            from CSE487.patching import patch_model_attention_with_mask
        patch_model_attention_with_mask(
            clip_masked,
            vision_layers,
            SmallWorldDebug,
            grid_shape=grid_shape,
            radius=radius,
            shortcut_prob=p,
            distance_type=distance_type,
            wrap_around=wrap_around,
            verbose=verbose  # Only print summary, not per-patch debug
        )
        acc = evaluate_clip_laion(clip_masked, processor, dataloader, device)
        for i, mg in enumerate(clip_masked.mask_generators):
            deg = mg.mask.sum(dim=1)
            print(f"[Layer {i}]  min={deg.min().item():3d}  max={deg.max().item():3d}  mean={deg.float().mean():.2f}")
            total_connections = mg.mask.sum().item()
            print(f"[Layer {i}] Total connections (attention patches): {total_connections}")
        print(f"→ Accuracy @ p={p:.2f}: {acc:.4f}")
        accuracies.append(acc)
    plt.figure(figsize=(8, 5))
    plt.plot(shortcut_probs, accuracies, marker='o', label="SmallWorldDebug Accuracy")
    plt.xlabel("shortcut_prob")
    plt.ylabel("Top-1 Accuracy")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run SmallWorldDebug mask experiment on CLIP or test mask standalone")
    parser.add_argument('--test_mask', action='store_true', help='Test mask generator standalone')
    parser.add_argument('--grid_shape', type=int, nargs=2, default=[14,14], help='Grid shape H W for mask')
    parser.add_argument('--radius', type=int, default=2, help='Radius for mask generator')
    parser.add_argument('--shortcut_prob', type=float, default=0.9, help='Shortcut probability for mask generator')
    parser.add_argument('--distance_type', type=str, default='manhattan', help="Distance type: 'manhattan' or 'euclidean'")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--shortcut_probs', type=float, nargs='+', default=[0.1])
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Standalone mask test
    if args.test_mask:
        H, W = args.grid_shape
        seq_len = H*W + 1
        mg = SmallWorldDebug(seq_len, device, grid_shape=(H,W), radius=args.radius,
                             shortcut_prob=args.shortcut_prob, distance_type=args.distance_type, wrap_around=True, verbose=False)
        mask = mg.get_mask()
        deg = mask.sum(dim=1).cpu().numpy()
        import numpy as np
        print(f"Mask standalone test: seq_len={seq_len}, grid_shape=({H},{W}), radius={args.radius}, p={args.shortcut_prob}, distance_type={args.distance_type}")
        print(f"Degree stats -> min: {deg.min()}, max: {deg.max()}, mean: {deg.mean():.2f}, std: {deg.std():.2f}")
        # Histogram
        counts = np.bincount(deg)
        for d, c in enumerate(counts):
            if c>0:
                print(f"  degree {d}: {c} nodes")
        import sys; sys.exit(0)

    # Data
    dataset = CLIPCaptionDataset("/home/thanhdo/CSE487/laion400m", "/home/thanhdo/CSE487/laion400m/captions.txt")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model/Processor
    from transformers import CLIPModel, CLIPProcessor
    clip_base = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    # Evaluate baseline (unmasked)
    print("\nEvaluating with NO MASK (baseline)...")
    baseline_acc = evaluate_clip_laion(clip_base, processor, dataloader, device)
    print(f"→ Baseline Accuracy: {baseline_acc:.4f}")
    # For baseline, total connections would be num_patches*num_patches, e.g., 197*197 = 38809 for ViT-B/16
    # This is different from the sparse connection counts printed for masked versions.

    rad_list = [2,4,8,16]
    for radius in rad_list:
        run_smallworld_debug_experiment(
            clip_base,
            processor,
            dataloader,
            device,
            shortcut_probs=args.shortcut_probs,
            grid_shape=tuple(args.grid_shape),
            radius=radius,
            distance_type=args.distance_type,
            wrap_around=True,
            verbose=False
        )
