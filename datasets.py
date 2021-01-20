import torch
import numpy as np
import random
from torch.utils.data import Dataset

random.seed(10)

class EchoKthNumber(Dataset):
    """Echo Kth Number dataset."""

    def __init__(self, size, kth_number, min_seq_len, max_seq_len):
        """
        Args:
            size (int): Desired size of dataset
            kth_number (int): Sequence element to echo
            min_seq_len (int): Minimum sequence length (>1)
            max_seq_len (int): Maximums sequence length (>min_seq_len)
        """
        assert 1 < min_seq_len <= max_seq_len
        self.size = size
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.k = kth_number - 1
        self._cache = {}

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        key = str(idx)
        if key in self._cache:
            return self._cache[key]
        seq_len = random.randint(self.min_seq_len, self.max_seq_len)
        x = torch.zeros(seq_len, 10)
        src = torch.ones(seq_len, 1)
        indices = [random.randint(0, 9) for _ in range(seq_len)]
        indices = torch.tensor(indices)
        indices.unsqueeze_(1)
        repr = indices
        x.scatter_(1, indices, src)
        y = indices[self.k]
        y = y.squeeze(0)
        self._cache[key] = (x, y)
        return x, y


class BinaryAddition(Dataset):
    def __init__(self, size, debug=False):
        self.size = size
        self.max = 63
        self.bits = 7
        self.fmt = '07b'
        self.debug = debug
        self._cache = {}

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        a_dig = random.randint(0, 63)
        b_dig = random.randint(0, 63)
        c_dig = a_dig + b_dig
        a = torch.tensor([float(x) for x in format(a_dig, self.fmt)][::-1])
        b = torch.tensor([float(x) for x in format(b_dig, self.fmt)][::-1])
        y = torch.tensor([int(x) for x in format(c_dig, self.fmt)][::-1])
        x = torch.stack((a, b), dim=-1)
        output = (x, y)
        self._cache[idx] = output
        if self.debug:
            print(f"Example {idx}")
            print(format(a_dig, self.fmt))
            print(format(b_dig, self.fmt))
            print('-' * 7)
            print(format(c_dig, self.fmt))
        return output







class CharacterDataset(Dataset):
    def __init__(self, filename='data/shakespeare/shakespeare.txt', seq_len=5, one_hot=True, debug=False):
        self.data = open(filename, 'r').read()
        cut_idx = (len(self.data) // 200) * 200
        self.data = self.data[:cut_idx]
        self.chars = list(set(self.data))
        self.data_size = len(self.data)
        self.vocab_size = len(self.chars)
        self._char_to_ix = {ch: i for i, ch in enumerate(self.chars)}
        self._ix_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.one_hot = one_hot
        self.debug = debug
        self.seq_len = seq_len

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        char = self.data[idx]
        x = self._char_to_ix[char]
        next_idx = (idx + 1) % self.data_size
        next_char = self.data[next_idx]
        y = torch.tensor(self._char_to_ix[next_char])
        if self.debug:
            print(f"{char} -> {next_char}")
        if self.one_hot:
            x = self._one_hot_encode(torch.tensor(x))
            # y = self._one_hot_encode(torch.tensor(y))
        return x, y

    def _one_hot_encode(self, seq):
        one_hot = torch.zeros(self.vocab_size)
        one_hot[seq] = 1.
        return one_hot

    @staticmethod
    def collate_fn(batch):
        xs = [b[0] for b in batch]
        ys = [b[1] for b in batch]
        x = torch.stack(xs, dim=0).reshape(40, 5, -1)
        y = torch.stack(ys, dim=0).reshape(40, 5)
        return x, y


