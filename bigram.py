"""This takes the simple intermediary work from gpt-dev.ipynb and converts it into a script

This is a simple bigram language model. It is a simple example of a language model that is trained to predict the next
character given the previous character.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F

"""Reproducibility"""
torch.manual_seed(1337)


"""Hyperparameters"""


batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'  # use the GPU if you have one
eval_iters = 200  # how many batches to average over when evaluating the loss
n_embd = 32  # Number of embedding dimensions


"""Data loading and preprocessing"""


# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()


# Extract the vocabulary
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s): return [stoi[c] for c in s]  # encoder: take a string, output a list of integers
def decode(l): return ''.join([itos[i] for i in l])  # decoder: take a list of integers, output a string


# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# Define a function to sample a batch of data


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


"""Define a loss function"""


@torch.no_grad()  # Tell PyTroch we wont call .backward() on this function (i.e. we don't need gradients)
def estimate_loss():
    """Average loss over multiple batches.

    Returns:
        out: dictionary with keys 'train' and 'val', each with a scalar value
    """
    out = {}
    model.eval()  # Put the model in evaluation mode
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()  # Get loss
        out[split] = losses.mean()  # Average loss
    model.train()  # Put the model back in training mode
    return out


"""A simple bigram language model"""


class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # Encode the tokens into embeddings
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  # Encode the position of the tokens
        self.lm_head = nn.Linear(n_embd, vocab_size)  # Linear model head: takes the embedding and outputs logits

    def forward(self, idx, targets=None):
        """Handls the forward pass of the model.

        Args:
            idx: (B,T) tensor of integers
            targets: (B,T) tensor of integers, or None

        Returns:
            logits: (B,T,C) tensor of logits
            loss: scalar tensor, or None
        """
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        token_embedding = self.token_embedding_table(idx)  # (B,T,C)
        positional_embedding = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)
        x = token_embedding + positional_embedding  # (B,T,C) - holds the token identitys and their position
        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        """Generates new tokens given a context.

        Args:
            idx: (B,T) tensor of integers
            max_new_tokens: int, maximum number of new tokens to generate

        Returns:
            idx: (B,T+1) tensor of integers
        """
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # get the predictions
            logits, loss = self(idx)
            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


"""Training loop"""
model = BigramLanguageModel()
m = model.to(device)  # If you have a GPU, this will use it

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


"""Display the model's predictions"""
# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
