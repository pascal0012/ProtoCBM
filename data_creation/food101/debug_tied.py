"""Is lm_head.weight actually tied to embed_tokens.weight after load?"""
import sys
sys.path.insert(0, "data_creation/food101")
import torch
from extract_descriptions import load_r4b

model, _, _ = load_r4b("YannQi/R-4B", "float32")
lm_head = model.lm_head.weight
embed = model.model.language_model.embed_tokens.weight
print(f"lm_head.weight:   shape={tuple(lm_head.shape)}, ptr={lm_head.data_ptr()}")
print(f"embed_tokens.w:   shape={tuple(embed.shape)}, ptr={embed.data_ptr()}")
print(f"same storage? {lm_head.data_ptr() == embed.data_ptr()}")
print(f"values equal?  {torch.equal(lm_head, embed)}")
print(f"lm_head norm:   {lm_head.norm().item():.4f}")
print(f"embed norm:     {embed.norm().item():.4f}")
print(f"lm_head[:3,:5]:\n{lm_head[:3,:5]}")
print(f"embed[:3,:5]:\n{embed[:3,:5]}")
