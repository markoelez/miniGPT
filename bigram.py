#!/usr/bin/env python3
import torch
import random
import torch.nn as nn
from torch.nn import functional as F


device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

torch.manual_seed(1337)

# hyperparameters
tts = 0.9
block_size = 8
batch_size = 32
steps = 20000
lr = 1e-2
n_eval = 200

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

dat = torch.tensor(encode(text), dtype=torch.long)

i = int(tts * len(dat))
train_dat, test_dat = dat[:i], dat[i:]

def get_batch(phase: str = 'train'):
  dat = train_dat if phase == 'train' else test_dat
  ix = torch.randint(len(dat) - block_size, (batch_size,))
  x = torch.stack([dat[i:i+block_size] for i in ix]).to(device)
  y = torch.stack([dat[i+1:i+block_size+1] for i in ix]).to(device)
  return x, y

@torch.no_grad()
def estimate_loss(model):
  out = {}
  model.eval()
  for split in ('train', 'val'):
    losses = torch.zeros(n_eval)
    for k in range(n_eval):
      xb, yb = get_batch()
      _, loss = model(xb, yb)
      losses[k] = loss
    out[split] = losses.mean()
  model.train()
  return out

class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

  def forward(self, idx, targets=None):
    logits = self.token_embedding_table(idx)
    if targets is None: return logits, None
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    targets = targets.view(B*T)
    loss = F.cross_entropy(logits, targets)
    return logits, loss

  def generate(self, idx, max_new_tokens):
    for _ in range(max_new_tokens):
      logits, _ = self(idx)
      logits = logits[:, -1, :]
      probs = F.softmax(logits, dim=-1)
      nx = torch.multinomial(probs, num_samples=1)
      idx = torch.cat((idx, nx), dim=1)
    return idx

  def sample(self, n: int = 1000):
    idx = torch.zeros((1, 1), dtype=torch.long).to(device)
    return decode(self.generate(idx, n)[0].tolist())


m = BigramLanguageModel(vocab_size).to(device)
optim = torch.optim.AdamW(m.parameters(), lr=lr)

for i in range(steps):
  if i % n_eval == 0:
    loss = estimate_loss(m)
    print(f'step {i}: train loss {loss["train"]:.4f}, val loss {loss["val"]:.4f}')

  xb, yb = get_batch()

  logits, loss = m(xb, yb)
  
  optim.zero_grad(set_to_none=True)
  loss.backward()
  optim.step()

print(m.sample())
