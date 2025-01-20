import torch

from utils import load_data, create_lookup_tables

# load data
data = load_data()
print(f'Total names: {len(data)}')

# create letters lookup tables
tokens_count = 27
stoi, itos = create_lookup_tables(data)

# calculate bi-grams' counts
bigrams = torch.zeros(tokens_count, tokens_count, dtype=torch.float)
for word in data:
    word = '.' + word + '.'
    for c1, c2 in zip(word, word[1:]):
        i1, i2 = stoi[c1], stoi[c2]
        bigrams[i1, i2] += 1

# train statistical model
bigrams += 1  # smooth model
probs = bigrams / bigrams.sum(1, keepdim=True)

# make predictions
gen = torch.Generator().manual_seed(42)
for i in range(10):
    res = []
    xi = 0
    while True:
        xi = torch.multinomial(probs[xi], 1, generator=gen).item()
        if xi == 0:
            break
        res.append(itos[xi])

    print(''.join(res))

# calculate loss
n = 0
log_likelihood = 0
for word in ['andrejq']:
    word = '.' + word + '.'
    for c1, c2 in zip(word, word[1:]):
        i1, i2 = stoi[c1], stoi[c2]
        log_proba = torch.log(probs[i1, i2])
        log_likelihood += log_proba
        n += 1

neg_log_likelihood = -log_likelihood / n
print(f'Negative log likelihood: {neg_log_likelihood}')
