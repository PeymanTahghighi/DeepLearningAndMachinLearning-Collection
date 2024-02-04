import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

block_size = 64;
batch_size = 32;
n_embed = 256;
max_iters = 5000;
learning_rate = 1e-4;
n_layers = 6;
n_head = 8;

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read();

chars = sorted(list(set(text)));
vocab_size = len(chars);

stoi = {ch:i for i, ch in enumerate(chars)};
itos = {i:ch for i, ch in enumerate(chars)};
encode = lambda s: [stoi[c] for c in s]
decode = lambda d: ''.join([itos[c] for c in d])

data = torch.tensor(encode(text), dtype=torch.long);

n = int(0.9*len(data));
train_data = data[:n];
val_data = data[n:];

def get_batch(split):
    data = train_data if split=='train' else val_data;
    ix = torch.randint(len(data) - block_size, (batch_size,));
    x = torch.stack([data[i:i+block_size] for i in ix]);
    y = torch.stack([data[i+1:i+block_size+1] for i in ix]);
    return x,y;

class Head(nn.Module):
    def __init__(self, head_size) -> None:
        super().__init__();
        self.key = nn.Linear(n_embed, head_size, bias = False);
        self.query = nn.Linear(n_embed, head_size, bias = False);
        self.value = nn.Linear(n_embed, head_size, bias = False);
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)));

    def forward(self, x):
        B,T,C = x.shape;
        k = self.key(x);
        q = self.query(x);
        scale = (k@q.transpose(-2,-1))*C**-0.5;
        scale = scale.masked_fill(self.tril[:T,:T] == 0, (float('-inf'))); #make a decoder block
        scale = F.softmax(scale, dim=-1);
        v = self.value(x);
        out = scale @ v;
        return out;

class MultiheadAttention(nn.Module):
    def __init__(self, num_heads, head_size) -> None:
        super().__init__();

        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)]);
        self.proj = nn.Linear(n_embed, n_embed);
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1);
        out = self.proj(out);
        return out;

class FeedForward(nn.Module):
    def __init__(self, n_embed):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, n_embed*4),
            nn.ReLU(),
            nn.Linear(n_embed*4, n_embed),
        )
    
    def forward(self, x):
        return self.net(x);

class Block(nn.Module):
    def __init__(self, n_embed, n_head) -> None:
        super().__init__()
        head_size = n_embed // n_head;
        self.sa = MultiheadAttention(n_head, head_size);
        self.ffwd = FeedForward(n_embed);
        self.ln1 = nn.LayerNorm(n_embed);
        self.ln2 = nn.LayerNorm(n_embed);
    def forward(self, x):
        x = self.ln1(x);
        x = x + self.sa(x);
        x = x + self.ffwd(self.ln2(x));
        return x;
 
class BigramLM(nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__();
        self.embedding = nn.Embedding(vocab_size, n_embed);
        self.position_embedding_table = nn.Embedding(block_size,n_embed);
        self.lm_head = nn.Linear(n_embed, vocab_size);
        self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layers)])
        self.ln = nn.LayerNorm(n_embed)


    def forward(self, idx, targets=None):
        B,T = idx.shape;
        tok_embedding = self.embedding(idx);
        pos_embed = self.position_embedding_table(torch.arange(T).to('cuda'))
        x = tok_embedding + pos_embed;
        x = self.blocks(x);
        x = self.ln(x);
        logits = self.lm_head(x);
        if targets is not None:
            loss = F.cross_entropy(logits.permute(0,2,1), targets)
            return logits, loss;
        return logits;

    def generate(self, idx, new_tokens):
        for _ in range(new_tokens):
            idx_cond = idx[:, -block_size:];
            logits = self(idx_cond);
            logits = logits[:,-1,:];
            probs = F.softmax(logits, dim = 1);
            id_next = torch.multinomial(probs, num_samples=1);
            idx = torch.cat((idx, id_next), dim=1);
        return idx;

B,T,C = 4,8,32
head_size = 16;
key = nn.Linear(C,head_size,bias=False);
query = nn.Linear(C,head_size,bias=False);


net = BigramLM(vocab_size).to('cuda');
optimzer = optim.AdamW(net.parameters(), lr = learning_rate);

for step in range(max_iters):
    xb, yb = get_batch('train');
    xb, yb = xb.to('cuda'), yb.to('cuda');
    logits, loss = net(xb,yb);
    loss.backward();
    optimzer.step();
    net.zero_grad(set_to_none=True);
    print(loss.item());

out = net.generate(torch.zeros((1,1), dtype=torch.long).to('cuda'),500)[0].tolist();
print(decode(out));