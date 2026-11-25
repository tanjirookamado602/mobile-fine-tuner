import os
import json
import numpy as np
import torch
import torch.nn as nn
import math

class Projector(nn.Module):
    def __init__(self, in_features=768, out_features=896, num_tokens=64, target_norm=0.4633):
        super().__init__()
        self.ln = nn.LayerNorm(in_features)
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.GELU(),
            nn.Linear(out_features, out_features)
        )
        init_scale = target_norm / math.sqrt(out_features)
        self.scale = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))
        init_std = target_norm / math.sqrt(out_features)
        self.pos_embed = nn.Parameter(torch.randn(1, num_tokens, out_features) * init_std)

    def forward(self, x):
        x = self.ln(x)
        out = self.net(x) * self.scale
        out = out + self.pos_embed
        return out

def main():
    out_dir = "data/parity_test"
    os.makedirs(out_dir, exist_ok=True)

    # 1. Initialize Projector with fixed seed
    torch.manual_seed(42)
    np.random.seed(42)
    proj = Projector(768, 896, 64)

    # Export initial weights
    weights = {
        "ln_w.bin": proj.ln.weight.detach().cpu().numpy().astype(np.float32),
        "ln_b.bin": proj.ln.bias.detach().cpu().numpy().astype(np.float32),
        "W1.bin": proj.net[0].weight.detach().cpu().numpy().astype(np.float32),
        "b1.bin": proj.net[0].bias.detach().cpu().numpy().astype(np.float32),
        "W2.bin": proj.net[2].weight.detach().cpu().numpy().astype(np.float32),
        "b2.bin": proj.net[2].bias.detach().cpu().numpy().astype(np.float32),
        "pos_embed.bin": proj.pos_embed.detach().cpu().numpy().astype(np.float32)
    }
    for name, arr in weights.items():
        arr.tofile(os.path.join(out_dir, name))
        print(f"Exported {name}: shape {arr.shape}, dtype {arr.dtype}")

    # 2. Export 1 Fixed Batch
    with open("data/multimodal_splits/train.jsonl") as f:
        sample = json.loads(f.readline())
    input_ids = np.array(sample["input_ids"], dtype=np.int32)
    labels = np.array(sample["labels"], dtype=np.int32)
    input_ids.tofile(os.path.join(out_dir, "input_ids.bin"))
    labels.tofile(os.path.join(out_dir, "labels.bin"))

    # Load 1 image feature
    feats = np.load("data/vision_features/visual_features.npy")
    # Take image 0: [64, 768]
    v_feat = feats[0].astype(np.float32)
    v_feat.tofile(os.path.join(out_dir, "visual_features.bin"))
    print(f"Exported batch: visual_features {v_feat.shape}, input_ids {input_ids.shape}, labels {labels.shape}")

    # 3. Load Qwen2.5-0.5B in PyTorch on CPU
    from transformers import AutoModelForCausalLM
    model_dir = "models/Qwen2.5-0.5B"
    print("Loading Qwen2.5-0.5B in PyTorch...")
    qwen = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True
    ).to("cpu")
    qwen.eval()
    for p in qwen.parameters():
        p.requires_grad = False

    # 4. Run 5 Training Steps in PyTorch with Adam
    optimizer = torch.optim.Adam(
        proj.parameters(),
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.0
    )

    losses = []
    v_tensor = torch.from_numpy(v_feat).unsqueeze(0) # [1, 64, 768]
    ids_tensor = torch.from_numpy(input_ids).unsqueeze(0).long()
    labels_tensor = torch.from_numpy(labels).unsqueeze(0).long()

    # Embedding lookup from Qwen
    text_embeds = qwen.model.embed_tokens(ids_tensor)

    print("\nRunning 5 Training Steps in PyTorch...")
    for step in range(1, 6):
        optimizer.zero_grad()
        p_tokens = proj(v_tensor)
        spliced = torch.cat([p_tokens, text_embeds], dim=1)

        # Forward through Qwen
        out = qwen.model(inputs_embeds=spliced)
        hidden = out.last_hidden_state
        logits = qwen.lm_head(hidden)

        # Shift logits and targets
        pred_logits = logits[:, :-1, :].contiguous()
        # prefix of 64 tokens masked with -100
        prefix_len = 64
        shift_labels = torch.full((1, pred_logits.shape[1]), -100, dtype=torch.long)
        for i in range(pred_logits.shape[1]):
            tidx = i - prefix_len + 1
            if 0 <= tidx < labels_tensor.shape[1]:
                shift_labels[0, i] = labels_tensor[0, tidx]

        loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
        loss = loss_fn(pred_logits.view(-1, pred_logits.size(-1)), shift_labels.view(-1))
        loss_val = loss.item()
        losses.append(loss_val)

        loss.backward()
        optimizer.step()

        print(f"  PyTorch Step {step}: Loss = {loss_val:.6f}")

    with open(os.path.join(out_dir, "pytorch_losses.json"), "w") as f:
        json.dump(losses, f, indent=2)
    print("PyTorch reference losses saved.")

if __name__ == "__main__":
    main()
