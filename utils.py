def load_data():
    with open('./names.txt') as f:
        data = f.read().splitlines()

    return data


def create_lookup_tables(data):
    letters = list(set(''.join(data)))
    stoi = {c: i + 1 for i, c in enumerate(sorted(letters))}
    stoi['.'] = 0
    itos = {i: c for c, i in stoi.items()}

    return stoi, itos
