import matplotlib.pyplot as plt
import numpy as np
from CSE487.mask_generators import BaseMaskGenerator, HubSpokeMask, PreferentialAttachmentMask, SmallWorldMask, InfoFlowHubSpoke, DynamicPreferentialAttachmentMask, LocalGraphMask, Debug

def plot_degree_stats(mask_generators, title_prefix=""):
    min_degrees, max_degrees, mean_degrees = [], [], []
    for mg in mask_generators:
        mask = mg.get_mask().cpu().numpy()
        degrees = mask.sum(axis=1)
        min_degrees.append(degrees.min())
        max_degrees.append(degrees.max())
        mean_degrees.append(degrees.mean())
    plt.figure(figsize=(10, 5))
    plt.plot(min_degrees, label="Min degree")
    plt.plot(max_degrees, label="Max degree")
    plt.plot(mean_degrees, label="Mean degree")
    plt.xlabel("Layer")
    plt.ylabel("Degree")
    plt.title(f"{title_prefix} Mask Degree Stats Across Layers")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_accuracy(acc_base, acc_masked, labels=("Baseline", "Masked")):
    plt.figure(figsize=(5, 5))
    plt.bar(labels, [acc_base, acc_masked], color=["skyblue", "salmon"])
    plt.ylabel("Accuracy")
    plt.title("CLIP Accuracy Comparison")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()
