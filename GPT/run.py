import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm

# Hyperparameters
block_size = 64  # Context window size
batch_size = 32  # Batch size
embed_size = 256  # Embedding dimension
max_iters = 100  # Number of training iterations
learning_rate = 1e-4  # Learning rate
n_layers = 12  # Number of transformer layers
n_head = 8  # Number of attention heads

# Load text data
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Create vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# Character mappings
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]  # Convert text to indices
decode = lambda d: ''.join([itos[c] for c in d])  # Convert indices to text

# Convert text data to tensor
data = torch.tensor(encode(text), dtype=torch.long)

# Train-validation split
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    """Generate a batch of training or validation data."""
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    return x, y

class Head(nn.Module):
    """Single attention head for multi-head self-attention."""
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embed_size, head_size, bias=False)
        self.query = nn.Linear(embed_size, head_size, bias=False)
        self.value = nn.Linear(embed_size, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        """Forward pass for the attention head."""
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        scale = (k @ q.transpose(-2, -1)) * C**-0.5
        scale = scale.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # Apply causal masking
        scale = F.softmax(scale, dim=-1)
        v = self.value(x)
        out = scale @ v
        return out

class MultiheadAttention(nn.Module):
    """Multi-head self-attention mechanism."""
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        """Forward pass for multi-head attention."""
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        return self.proj(out)

class FeedForward(nn.Module):
    """Feed-forward network used in the transformer block."""
    def __init__(self, embed_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.ReLU(),
            nn.Linear(embed_size * 4, embed_size),
        )
    
    def forward(self, x):
        """Forward pass for the feed-forward network."""
        return self.net(x)

class Block(nn.Module):
    """Transformer block with self-attention and feed-forward layers."""
    def __init__(self, embed_size, n_head):
        super().__init__()
        head_size = embed_size // n_head
        self.sa = MultiheadAttention(n_head, head_size)
        self.ffwd = FeedForward(embed_size)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)
    
    def forward(self, x):
        """Forward pass for the transformer block."""
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
 
class NanoGPT(nn.Module):
    """Minimal GPT-style transformer model."""
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.position_embedding_table = nn.Embedding(block_size, embed_size)
        self.lm_head = nn.Linear(embed_size, vocab_size)
        self.blocks = nn.Sequential(*[Block(embed_size, n_head=n_head) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(embed_size)

    def forward(self, idx, targets=None):
        """Forward pass for the model."""
        B, T = idx.shape
        tok_embedding = self.embedding(idx)
        pos_embed = self.position_embedding_table(torch.arange(T).to('cuda'))
        x = tok_embedding + pos_embed
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)
        
        if targets is not None:
            loss = F.cross_entropy(logits.permute(0, 2, 1), targets)
            return logits, loss
        return logits

    def generate(self, idx, new_tokens):
        """Generate new text based on input context."""
        for _ in range(new_tokens):
            idx_cond = idx[:, -block_size:]
            logits = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=1)
            id_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, id_next), dim=1)
        return idx

def train(net, optimizer):
    """Train the NanoGPT model."""
    for step in tqdm(range(max_iters)):
        xb, yb = get_batch('train')
        xb, yb = xb.to('cuda'), yb.to('cuda')
        logits, loss = net(xb, yb)
        loss.backward()
        optimizer.step()
        net.zero_grad(set_to_none=True)
        print(loss.item())
    torch.save(net.state_dict(), 'model.pt')

def test(net):
    """Test the trained NanoGPT model."""
    ckpt = torch.load('model.pt')
    net.load_state_dict(ckpt)
    with torch.no_grad():
        net.eval()
        out = net.generate(torch.zeros((1, 1), dtype=torch.long).to('cuda'), 500)[0].tolist()
        print(decode(out))

if __name__ == "__main__":
    net = NanoGPT(vocab_size).to('cuda')
    optimizer = optim.AdamW(net.parameters(), lr=learning_rate)
    train(net, optimizer)
    test(net)
