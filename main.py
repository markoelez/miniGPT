#!/usr/bin/env python3
import torch
import random
from tqdm import tqdm
import torch.nn as nn
from torch.nn import functional as F


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
#device = 'cuda' if torch.cuda.is_available() else 'cpu'

torch.manual_seed(1337)

# hyperparameters
tts = 0.9
block_size = 256
batch_size = 128
steps = 20000
eval_interval = 400
n_eval = 200
lr = 3e-4
n_embed = 384
n_layer = 12
n_head = 8
dropout = 0.2

def load(fname):
  with open('data.txt', 'r') as f:
    return f.read()

text = load('data.txt')

vocab = sorted(set(text))

vocab_size = len(vocab)

stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}

def encode(s: str) -> list[int]:
  return [stoi[x] for x in s]

def decode(a: list[int]) -> str:
  return ''.join([itos[x] for x in a])

dat = torch.tensor(encode(text), dtype=torch.long).to(device)

i = int(tts * len(dat))
train_dat, test_dat = dat[:i], dat[i:]

def get_batch(phase: str = 'train'):
  dat = train_dat if phase == 'train' else test_dat
  ix = torch.randint(len(dat) - block_size, (batch_size,)).to(device)
  x = torch.stack([dat[i:i+block_size] for i in ix]).to(device)
  y = torch.stack([dat[i+1:i+block_size+1] for i in ix]).to(device)
  return x, y

@torch.no_grad()
def estimate_loss(model):
  out = {}
  model.eval()
  for split in ('train', 'val'):
    losses = torch.zeros(n_eval, device=device)
    for k in range(n_eval):
      xb, yb = get_batch()
      _, loss = model(xb, yb)
      losses[k] = loss
    out[split] = losses.mean()
  model.train()
  return out

class Head(nn.Module):
  """one head of self attention"""
  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embed, head_size, bias=False)
    self.query = nn.Linear(n_embed, head_size, bias=False)
    self.value = nn.Linear(n_embed, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B, T, C = x.shape

    k = self.key(x) # (B, T, head_size)
    q = self.query(x) # (B, T, head_size)

    # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
    wei = q @ k.transpose(-2, -1) * C**-0.5

    wei = wei.masked_fill(self.tril[:T,:T] == 0, float('-inf'))
    wei = F.softmax(wei, dim=-1)
    wei = self.dropout(wei)

    v = self.value(x)
    out = wei @ v

    return out

class MultiHeadAttention(nn.Module):
  """multiple heads of self attention"""
  def __init__(self, n_head, head_size):
    super().__init__()
    self.heads = nn.ModuleList([Head(head_size) for _ in range(n_head)])
    self.proj = nn.Linear(n_embed, n_embed)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out)
    out = self.dropout(out)
    return out

class FeedForward(nn.Module):
  def __init__(self):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embed, 4 * n_embed),
      nn.ReLU(),
      nn.Linear(4 * n_embed, n_embed),
      nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)

class Block(nn.Module):
  def __init__(self, n_embed, n_head):
    super().__init__()
    head_size = n_embed // n_head
    self.sa = MultiHeadAttention(n_head, head_size)
    self.ffwd = FeedForward()
    self.ln1 = nn.LayerNorm(n_embed)
    self.ln2 = nn.LayerNorm(n_embed)

  def forward(self, x):
    x = x + self.sa(self.ln1(x))
    x = x + self.ffwd(self.ln2(x))
    return x

class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.tok_embedding_table = nn.Embedding(vocab_size, n_embed)
    self.pos_embedding_table = nn.Embedding(block_size, n_embed)
    self.blocks = nn.Sequential(*[Block(n_embed, n_head=n_head) for _ in range(n_layer)])
    self.ln_f = nn.LayerNorm(n_embed)
    self.lm_head = nn.Linear(n_embed, vocab_size)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    tok_emb = self.tok_embedding_table(idx)
    pos_emb = self.pos_embedding_table(torch.arange(T, device=device))

    x = tok_emb + pos_emb
    x = self.blocks(x)
    x = self.ln_f(x)
    logits = self.lm_head(x) 

    if targets is None: return logits, None

    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = targets.view(B*T)
    loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      idx_cond = idx[:, -block_size:]
      logits, _ = self(idx_cond)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      nx = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, nx), dim=1)
    return idx

  def sample(self, n: int = 1000):
    idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    return decode(self.generate(idx, n)[0].tolist())

if __name__ == '__main__':

  model = BigramLanguageModel()
  m = model.to(device)
  
  optim = torch.optim.AdamW(model.parameters(), lr=lr)

  pbar = tqdm(range(steps), desc='Train', ncols=100)

  for i in pbar:
    if i % eval_interval == 0 or i == steps - 1:
      loss = estimate_loss(model)
      pbar.set_postfix({
        'train_loss': f'{loss["train"]:.4f}',
        'val_loss': f'{loss["val"]:.4f}'
      })

    xb, yb = get_batch()

    logits, loss = model(xb, yb)
    
    optim.zero_grad(set_to_none=True)
    loss.backward()
    optim.step()

  print()
  print(model.sample())

  torch.save(model.state_dict(), 'model.pth')
