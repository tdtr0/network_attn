import types
import torch
from torch import nn, Tensor
from mask_generators import BaseMaskGenerator, HubSpokeMask, PreferentialAttachmentMask, SmallWorldMask, InfoFlowHubSpoke, DynamicPreferentialAttachmentMask, LocalGraphMask, Debug

def patch_model_attention_with_mask(model, attention_layers, mask_generator_class, **mask_kwargs):
    """
    Patches each SelfAttention module in `attention_layers` so that:
      - Exactly one mask-generator instance is attached per layer.
      - That instance’s seq_len/device are updated as needed, without re-init.
      - model.mask_generators is a list of those instances.
      - Each mask_generator must have:
            get_mask() -> torch.BoolTensor [seq_len, seq_len]
            get_stats() -> dict with total_connections, local_connections, shortcut_connections
    """
    model.mask_generators = []
    for layer_idx, layer in enumerate(attention_layers):
        attn = layer.self_attn
        device = next(attn.parameters()).device
        mg = mask_generator_class(seq_len=0, device=device, **mask_kwargs)
        model.mask_generators.append(mg)
        def make_new_forward(mask_generator):
            def new_forward(self, hidden_states, attention_mask=None, causal_attention_mask=None, output_attentions=False):
                batch_size, seq_len, _ = hidden_states.shape
                dev = hidden_states.device
                if mask_generator.seq_len != seq_len or mask_generator.device != dev:
                    mask_generator.seq_len = seq_len
                    mask_generator.device = dev
                mask = mask_generator.get_mask()
                query = self.q_proj(hidden_states)
                key   = self.k_proj(hidden_states)
                value = self.v_proj(hidden_states)
                query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                key   = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                mask_exp = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, self.num_heads, seq_len, seq_len)
                scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
                scores = scores.masked_fill(~mask_exp, float('-inf'))
                probs = nn.functional.softmax(scores, dim=-1)
                probs = nn.functional.dropout(probs, p=self.dropout, training=self.training)
                out = torch.matmul(probs, value)
                out = out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
                return self.out_proj(out), None
            return new_forward
        attn.forward = types.MethodType(make_new_forward(mg), attn)
    print(f"Patched {len(attention_layers)} attention layers with dynamic masks.")

def patch_model_attention_with_sparse(model, attention_layers, mask_generator_class, **mask_kwargs):
    for layer_idx, layer in enumerate(attention_layers):
        attn_module = layer.self_attn
        def make_new_forward(layer_idx):
            @torch.compile(dynamic=True)
            def new_forward(self, hidden_states, attention_mask=None, causal_attention_mask=None, output_attentions=False):
                batch_size, seq_len, _ = hidden_states.shape
                device = hidden_states.device
                mask_generator = mask_generator_class(seq_len, device, layer_idx=layer_idx, **mask_kwargs)
                mask = mask_generator.get_mask()  # [seq_len, seq_len] bool
                query = self.q_proj(hidden_states)
                key   = self.k_proj(hidden_states)
                value = self.v_proj(hidden_states)
                query = query.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                key   = key.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                value = value.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                attn_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
                attn_scores = attn_scores.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float('-inf'))
                attn_probs = F.softmax(attn_scores, dim=-1)
                attn_probs = F.dropout(attn_probs, p=self.dropout, training=self.training)
                attn_output = torch.matmul(attn_probs, value)
                attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)
                return self.out_proj(attn_output), None
            return new_forward
        attn_module.forward = types.MethodType(make_new_forward(layer_idx), attn_module)
    print(f"Patched {len(attention_layers)} attention layers with dynamic sparse masking (torch.compile).")
