import torch
from tqdm import tqdm
from CSE487.mask_generators import BaseMaskGenerator, HubSpokeMask, PreferentialAttachmentMask, SmallWorldMask, InfoFlowHubSpoke, DynamicPreferentialAttachmentMask, LocalGraphMask, Debug
from CSE487.patching import patch_model_attention_with_mask

def evaluate_clip_laion(model, processor, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader):
            images = batch["image"]
            texts = batch["text"]
            # Encode using processor
            inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image  # [batch_size x batch_size]
            preds = logits_per_image.argmax(dim=-1)      # best-matching text for each image
            ground_truth = torch.arange(len(images), device=device)
            correct += (preds == ground_truth).sum().item()
            total += len(images)
    acc = correct / total
    return acc

def run_clip_experiment(clip_base, clip_masked, processor, dataloader, device):
    acc_base = evaluate_clip_laion(clip_base, processor, dataloader, device)
    acc_masked = evaluate_clip_laion(clip_masked, processor, dataloader, device)
    print(f"\nBaseline CLIP Accuracy: {acc_base:.4f}")
    print(f"Masked CLIP Accuracy: {acc_masked:.4f}")
    return acc_base, acc_masked

def print_mask_stats(model):
    for i, mg in enumerate(model.mask_generators):
        stats = mg.get_stats()
        print(f"Layer {i}: total={stats['total_connections']}, local={stats['local_connections']}, shortcut={stats['shortcut_connections']}")
