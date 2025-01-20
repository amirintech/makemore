import torch
import torch.nn.functional as F

from utils import load_data, create_lookup_tables

# load data
data = load_data()
print(f'Total names: {len(data)}')

# create letters lookup tables
tokens_count = 27
stoi, itos = create_lookup_tables(data)

# create training data
x, y = [], []
for word in data:
    word = '.' + word + '.'
    for c1, c2 in zip(word, word[1:]):
        i1, i2 = stoi[c1], stoi[c2]
        x.append(i1)
        y.append(i2)

x = torch.tensor(x, dtype=torch.long)
y = torch.tensor(y, dtype=torch.long)

# initialize weights
gen = torch.Generator().manual_seed(42)
W = torch.randn((tokens_count, tokens_count), requires_grad=True, generator=gen)

# train model
learning_rate = 30
epochs = 100
for e in range(epochs):
    # forward pass
    x_encoded = F.one_hot(x, num_classes=tokens_count).float()
    logits = x_encoded @ W
    counts = logits.exp()
    proba = counts / counts.sum(1, keepdim=True)
    loss = -proba[torch.arange(len(x)), y].log().mean()

    # backward pass
    W.grad = None
    loss.backward()

    if e % 10 == 0:
        print(f'epoch: {e}\tloss: {loss.item()}')

    # update weights
    W.data += -learning_rate * W.grad

# make predictions
for i in range(10):
    res = []
    xi = 0
    while True:
        x_encoded = F.one_hot(torch.tensor([xi]), num_classes=tokens_count).float()
        logits = x_encoded @ W
        counts = logits.exp()
        proba = counts / counts.sum(1, keepdim=True)
        xi = torch.multinomial(proba, 1, replacement=True).item()
        if xi == 0:
            break

        res.append(itos[xi])

    print(''.join(res))
