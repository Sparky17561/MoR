import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import math

# ---- Training Data ----
text = """Goldfish are a species of freshwater fish commonly kept as pets in aquariums and ponds. They belong to the carp family and are native to East Asia, where they were domesticated over a thousand years ago. Goldfish come in a wide variety of shapes, sizes, and colors, ranging from the classic orange to white, black, and calico patterns. They are known for their social behavior and can recognize their owners over time. Goldfish are omnivorous, meaning they eat both plant matter and small animals. Their diet usually consists of algae, small insects, larvae, and specially formulated fish food provided by humans. Proper care of goldfish requires a clean tank, good water filtration, and adequate space, since they can grow much larger than most people expect. Goldfish can live for more than ten years in healthy environments, with some reaching over twenty years of age. Because of their history, cultural significance, and hardy nature, goldfish are one of the most popular and recognizable aquarium fish in the world.
"""

# ---- Word-level Tokenizer (with special tokens) ----
words = text.lower().split()
unique_words = sorted(set(words))

# special tokens: PAD and UNK optional; we must include EOS
specials = ["<PAD>", "<UNK>", "<EOS>"]
vocab = specials + unique_words

stoi = {w: i for i, w in enumerate(vocab)}
itos = {i: w for i, w in enumerate(vocab)}

PAD_ID = stoi["<PAD>"]
UNK_ID = stoi["<UNK>"]
EOS_ID = stoi["<EOS>"]

def encode(s):
    ids = []
    for w in str(s).lower().split():
        ids.append(stoi.get(w, UNK_ID))
    return ids

def decode(ids):
    words_out = []
    for i in ids:
        if i == EOS_ID:
            break
        # guard
        if i < 0 or i >= len(itos):
            words_out.append("<UNK>")
        else:
            words_out.append(itos[i])
    return " ".join(words_out)

# create token sequence for training; append an EOS token at the end
data = torch.tensor(encode(text) + [EOS_ID], dtype=torch.long)

# ---- Dataset ----
class WordDataset(Dataset):
    def __init__(self, data, block_size=8):
        self.data = data
        self.block_size = block_size
    def __len__(self):
        return len(self.data) - self.block_size
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y

dataset = WordDataset(data, block_size=8)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# ---- Mixture of Recursions Block ----
class MoRBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=2, inner_layers=2):
        super().__init__()
        # note: MultiheadAttention expects (seq, batch, embed_dim) by default
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.ReLU(),
            nn.Linear(embed_dim*4, embed_dim)
        )
        self.inner_layers = inner_layers
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x shape: (seq, batch, dim)
        for _ in range(self.inner_layers):
            attn_out, _ = self.attn(x, x, x)
            x = self.norm1(x + attn_out)
            ff_out = self.ff(x)
            x = self.norm2(x + ff_out)
        return x

# ---- MoR Language Model ----
class MoRLM(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, num_layers=2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.blocks = nn.ModuleList([MoRBlock(embed_dim) for _ in range(num_layers)])
        self.ln = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x: (batch, seq)
        # convert to (seq, batch, dim) for MultiheadAttention
        x = self.embed(x).transpose(0,1)  # (seq, batch, dim)
        for block in self.blocks:
            x = block(x)
        x = self.ln(x)
        x = self.fc(x).transpose(0,1)  # (batch, seq, vocab)
        return x

# ---- Training ----
device = "cpu"
model = MoRLM(vocab_size=len(vocab)).to(device)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

epochs = 10
for epoch in range(epochs):
    loop = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
    for xb, yb in loop:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        # use reshape instead of view for non-contiguous tensors
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), yb.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        loop.set_postfix(loss=loss.item())

# ---- Improved Generation ----
def top_k_top_p_filtering(logits, top_k=0, top_p=0.0):
    """
    Filter a distribution of logits using top-k and/or nucleus (top-p) filtering.
    Accepts logits shape (V,) or (B, V). Returns same shape as input.
    """
    is_1d = False
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)   # (1, V)
        is_1d = True

    logits = logits.clone()
    batch_size, vocab_size = logits.shape

    for b in range(batch_size):
        row = logits[b]
        # top-k
        if top_k > 0:
            k = min(max(top_k, 1), vocab_size)
            # get kth highest value
            topk_vals = torch.topk(row, k).values
            kth_val = topk_vals[-1]
            row[row < kth_val] = -float("Inf")

        # top-p (nucleus)
        if top_p is not None and top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(row, descending=True)
            probs = torch.softmax(sorted_logits, dim=-1)
            cumulative_probs = torch.cumsum(probs, dim=-1)

            # mask tokens with cumulative prob above top_p
            sorted_indices_to_remove = cumulative_probs > top_p
            # shift right to keep first token above threshold
            if sorted_indices_to_remove.numel() > 1:
                sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
                sorted_indices_to_remove[0] = False

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            if indices_to_remove.numel() > 0:
                row[indices_to_remove] = -float("Inf")

        logits[b] = row

    return logits[0] if is_1d else logits


def generate(prompt,
             max_new_tokens=50,
             temperature=0.8,
             top_k=50,
             top_p=0.9,
             repetition_penalty=1.2,
             frequency_penalty=0.5,
             block_ngram_repeat=3,
             eos_token_id=EOS_ID,
             confidence_threshold=0.02):
    """
    Improved generate:
    - returns only the new continuation (not the prompt)
    - applies repetition & frequency penalties
    - blocks repeating n-grams of length block_ngram_repeat
    - uses top-k + top-p + temperature sampling
    - returns \"I don't know.\" if the model's top token prob < confidence_threshold
    """
    model.eval()
    context_ids = encode(prompt)
    input_ids = torch.tensor([context_ids], dtype=torch.long, device=device)  # (1, T)
    generated = []  # newly generated token ids (not including prompt)
    # history of ngrams seen in generated (for blocking)
    ngram_set = set()

    for step in range(max_new_tokens):
        logits = model(input_ids)  # (1, T, V)
        next_logits = logits[:, -1, :].squeeze(0).float()  # (V,)

        # apply repetition_penalty and frequency penalty based on generated tokens
        if len(generated) > 0:
            token_counts = {}
            for t in generated:
                token_counts[t] = token_counts.get(t, 0) + 1
            # frequency penalty: subtract freq*penalty from logits
            for token_id, count in token_counts.items():
                next_logits[token_id] = next_logits[token_id] - frequency_penalty * count
            # repetition penalty: divide logits for tokens already generated
            for token_id in set(generated):
                if repetition_penalty > 1.0:
                    next_logits[token_id] = next_logits[token_id] / repetition_penalty

        # n-gram blocking: ban any token that would create an ngram already in generated
        if block_ngram_repeat > 0 and len(generated) >= (block_ngram_repeat - 1):
            # build last (n-1) tokens
            last_ng_prefix = tuple(generated[-(block_ngram_repeat-1):]) if block_ngram_repeat > 1 else tuple()
            # compute which tokens would create a seen ngram
            banned = set()
            # iterate all vocab tokens (vocab is small here)
            for cand in range(len(vocab)):
                candidate_ngram = tuple(list(last_ng_prefix) + [cand])
                if candidate_ngram in ngram_set:
                    banned.add(cand)
            if banned:
                for b in banned:
                    next_logits[b] = -float("Inf")

        # apply temperature + top_k/top_p filtering
        if temperature != 1.0 and temperature > 0.0:
            scaled = next_logits / temperature
        else:
            scaled = next_logits

        scaled = top_k_top_p_filtering(scaled.unsqueeze(0), top_k=top_k, top_p=top_p).squeeze(0)

        probs = torch.softmax(scaled, dim=-1)

        # safety: if top token prob is too low, abstain
        top_prob, top_idx = torch.max(probs, dim=-1)
        if top_prob.item() < confidence_threshold:
            return "I don't know."

        next_token_id = int(torch.multinomial(probs, num_samples=1).item())

        # append and update structures
        generated.append(next_token_id)
        # update ngram_set with any new ngrams that appear (we only track ngrams contained purely inside generated)
        if block_ngram_repeat > 0 and len(generated) >= block_ngram_repeat:
            # add the last ngram
            ng = tuple(generated[-block_ngram_repeat:])
            ngram_set.add(ng)

        # append token to input_ids for autoregressive step
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]], device=device)], dim=1)

        if eos_token_id is not None and next_token_id == eos_token_id:
            break

        # additional guard: avoid long exact repetitions of the same token
        if len(generated) >= 10:
            last10 = generated[-10:]
            if all(x == last10[0] for x in last10):
                break

    # decode only the generated tokens (not the prompt)
    return decode(generated)

# ---- Try Asking a Question ----
print("Prompt -> Response")
print("-------------------")
print("Q: Which family do goldfish belong to?")
print("A:", generate("which family goldfish belong to", max_new_tokens=60,
                   temperature=0.6, top_k=40, top_p=0.9,
                   repetition_penalty=1.3, frequency_penalty=0.6, block_ngram_repeat=3))
