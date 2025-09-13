# mor_model.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# -------------------------
# Word-level Tokenizer
# -------------------------
class WordTokenizer:
    def __init__(self, vocab_file):
        self.itos = []
        with open(vocab_file, "r", encoding="utf-8") as f:
            for line in f:
                self.itos.append(line.strip())
        self.stoi = {w: i for i, w in enumerate(self.itos)}
        # special tokens (must exist in vocab file)
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
        # ids (fall back to sensible defaults if missing)
        self.pad_id = self.stoi.get(self.pad_token, 0)
        self.unk_id = self.stoi.get(self.unk_token, 1)
        self.bos_id = self.stoi.get(self.bos_token, None)
        self.eos_id = self.stoi.get(self.eos_token, None)

    def encode(self, text):
        """Whitespace tokenization, lowercased. Returns list[int]."""
        tokens = []
        for w in str(text).lower().split():
            tokens.append(self.stoi.get(w, self.unk_id))
        return tokens

    def decode(self, ids):
        """Decode token ids to a string until EOS (if present)."""
        words = []
        for i in ids:
            if i == self.eos_id:
                break
            # protect against out-of-range ids
            if i < 0 or i >= len(self.itos):
                words.append(self.itos[self.unk_id])
            else:
                words.append(self.itos[i])
        return " ".join(words)

# -------------------------
# Small Transformer pieces (causal)
# -------------------------
def causal_mask(sz, device):
    # attn_mask additive form: shape (T, T), upper triangular set to -inf
    mask = torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)
    return mask

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_hidden):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.ln1 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim)
        )
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, x, attn_mask=None):
        # x: (B, T, D)
        residual = x
        attn_out, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = self.ln1(residual + attn_out)
        residual = x
        x = self.ln2(residual + self.ff(x))
        return x

# -------------------------
# Mixture-of-Recursions LM (causal)
# -------------------------
class MoRLM(nn.Module):
    def __init__(self, vocab_size, dim=256, n_heads=4, mlp_hidden=512, max_depth=4, halting_eps=0.99, max_pos=1024):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_pos, dim)
        self.shared_block = TransformerBlock(dim, n_heads, mlp_hidden)
        # halting unit for ACT-style halting
        self.halting_proj = nn.Linear(dim, 1)
        self.max_depth = max_depth
        self.halting_eps = halting_eps
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, input_ids):
        # input_ids: (B, T)
        B, T = input_ids.shape
        device = input_ids.device
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, T)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        halting_probability = torch.zeros(B, T, device=device)
        n_updates = torch.zeros(B, T, device=device)
        still_running = torch.ones(B, T, dtype=torch.bool, device=device)
        accumulated_state = torch.zeros_like(x)

        attn_mask = causal_mask(T, device)

        step = 0
        while step < self.max_depth and still_running.any():
            x = self.shared_block(x, attn_mask=attn_mask)
            p = torch.sigmoid(self.halting_proj(x)).squeeze(-1)  # (B,T)
            add = torch.where(still_running, p, torch.zeros_like(p))
            new_halted = (halting_probability + add > self.halting_eps) & still_running
            add_prob = torch.where(new_halted, (self.halting_eps - halting_probability).clamp(min=0.0), add)
            halting_probability = halting_probability + add_prob
            accumulated_state = accumulated_state + x * add_prob.unsqueeze(-1)
            n_updates = n_updates + still_running.float()
            still_running = still_running & (~new_halted)
            step += 1

        # leftover mass
        remaining = (1.0 - halting_probability).clamp(min=0.0)
        accumulated_state = accumulated_state + x * remaining.unsqueeze(-1)

        final = self.ln(accumulated_state)
        logits = self.head(final)  # (B, T, V)
        return logits

    # generation: greedy / sampling with EOS stop
    @torch.no_grad()
    def generate(self, tokenizer, prompt_text, max_new_tokens=64, device=None, temperature=1.0, top_k=0):
        if device is None:
            device = next(self.parameters()).device
        self.eval()
        ids = tokenizer.encode(prompt_text)
        if tokenizer.bos_id is not None:
            ids = [tokenizer.bos_id] + ids
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            logits = self.forward(input_ids)  # (1, T, V)
            next_logits = logits[:, -1, :]  # (1, V)

            # temperature / greedy handling
            if temperature <= 0.0:
                # greedy
                next_token_id = int(torch.argmax(next_logits, dim=-1)[0].item())
            else:
                scaled = next_logits / temperature

                if top_k and top_k > 0:
                    # set all but top_k logits to -inf
                    values, indices = torch.topk(scaled, top_k, dim=-1)
                    mask = torch.full_like(scaled, float("-inf"))
                    mask.scatter_(1, indices, values)
                    scaled = mask

                probs = torch.softmax(scaled, dim=-1)
                next_token_id = int(torch.multinomial(probs[0], num_samples=1).item())

            input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)

            if tokenizer.eos_id is not None and next_token_id == tokenizer.eos_id:
                break

        output_ids = input_ids[0].tolist()
        if tokenizer.bos_id is not None and output_ids and output_ids[0] == tokenizer.bos_id:
            output_ids = output_ids[1:]
        return tokenizer.decode(output_ids)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save({
            'state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'dim': self.dim
        }, path)

    @classmethod
    def load(cls, path, device="cpu"):
        ck = torch.load(path, map_location=device)
        vocab_size = ck.get('vocab_size')
        dim = ck.get('dim')
        model = cls(vocab_size=vocab_size, dim=dim)
        model.load_state_dict(ck['state_dict'])
        model.to(device)
        return model
