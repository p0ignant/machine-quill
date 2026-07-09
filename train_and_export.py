"""
train_and_export.py — Trains a small character-level GPT and exports its weights
so they can run in a browser via pure JavaScript (no server, no Python needed
once deployed — perfect for GitHub Pages).

Run locally (not in a notebook without a GPU/CPU budget in mind):
    pip install torch numpy
    python train_and_export.py

Output (put these two files in your web/ folder before deploying):
    web/weights.bin   — all model weights, packed as raw float32
    web/meta.json     — vocabulary + architecture info the JS needs to rebuild the model
"""

import os
import json
import struct
import urllib.request

import torch
import torch.nn as nn
from torch.nn import functional as F

# -----------------------------------------------------------------------------
# CONFIG — kept small on purpose so it downloads/runs fast in a browser.
# Feel free to bump these up if you don't mind a bigger weights.bin file.
# -----------------------------------------------------------------------------

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = "input.txt"

BATCH_SIZE = 64
BLOCK_SIZE = 64        # context length — smaller than before, to keep the JS demo fast
N_EMBED = 128
N_HEAD = 4
N_LAYER = 4
DROPOUT = 0.1

MAX_ITERS = 5000
EVAL_INTERVAL = 500
EVAL_ITERS = 50
LEARNING_RATE = 3e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)


def load_data():
    if not os.path.exists(DATA_PATH):
        print(f"Downloading training data from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)

    with open(DATA_PATH, "r", encoding="utf-8") as f:
        text = f.read()

    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)

    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    return data[:n], data[n:], vocab_size, stoi, itos, encode, decode


def get_batch(split, train_data, val_data):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCK_SIZE, (BATCH_SIZE,))
    x = torch.stack([data[i : i + BLOCK_SIZE] for i in ix])
    y = torch.stack([data[i + 1 : i + BLOCK_SIZE + 1] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)


class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(N_EMBED, head_size, bias=False)
        self.query = nn.Linear(N_EMBED, head_size, bias=False)
        self.value = nn.Linear(N_EMBED, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE)))
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * (C ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        return wei @ v


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(N_EMBED, N_EMBED)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, 4 * n_embed),
            nn.GELU(),
            nn.Linear(4 * n_embed, n_embed),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, n_embed, n_head):
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, N_EMBED)
        self.position_embedding = nn.Embedding(BLOCK_SIZE, N_EMBED)
        self.blocks = nn.Sequential(*[Block(N_EMBED, N_HEAD) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBED)
        self.lm_head = nn.Linear(N_EMBED, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            B, T, V = logits.shape
            loss = F.cross_entropy(logits.view(B * T, V), targets.view(B * T))
        return logits, loss


@torch.no_grad()
def estimate_loss(model, train_data, val_data):
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = torch.zeros(EVAL_ITERS)
        for k in range(EVAL_ITERS):
            X, Y = get_batch(split, train_data, val_data)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def export_for_web(model, stoi, itos, vocab_size, out_dir="web"):
    """Writes weights.bin (raw float32 weights, concatenated) and meta.json
    (vocab + shapes + byte offsets) so model.js can reconstruct the model."""
    os.makedirs(out_dir, exist_ok=True)

    state_dict = model.state_dict()
    buffer = bytearray()
    tensor_info = {}

    for name, tensor in state_dict.items():
        flat = tensor.detach().cpu().numpy().astype("float32").flatten()
        offset = len(buffer)
        buffer.extend(struct.pack(f"{len(flat)}f", *flat))
        tensor_info[name] = {
            "shape": list(tensor.shape),
            "offset": offset,
            "length": len(flat),
        }

    with open(os.path.join(out_dir, "weights.bin"), "wb") as f:
        f.write(buffer)

    meta = {
        "vocab_size": vocab_size,
        "stoi": stoi,
        "itos": {str(k): v for k, v in itos.items()},
        "block_size": BLOCK_SIZE,
        "n_embed": N_EMBED,
        "n_head": N_HEAD,
        "n_layer": N_LAYER,
        "tensors": tensor_info,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f)

    size_mb = len(buffer) / (1024 * 1024)
    print(f"\nExported to {out_dir}/weights.bin ({size_mb:.2f} MB) and {out_dir}/meta.json")


def main():
    print(f"Using device: {DEVICE}")
    train_data, val_data, vocab_size, stoi, itos, encode, decode = load_data()
    print(f"Vocabulary size: {vocab_size}")

    model = GPT(vocab_size).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_params / 1e6:.2f}M parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    for it in range(MAX_ITERS):
        if it % EVAL_INTERVAL == 0 or it == MAX_ITERS - 1:
            losses = estimate_loss(model, train_data, val_data)
            print(f"step {it:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")

        xb, yb = get_batch("train", train_data, val_data)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    print("\nTraining complete.")
    model.eval()
    export_for_web(model, stoi, itos, vocab_size, out_dir="web")


if __name__ == "__main__":
    main()
