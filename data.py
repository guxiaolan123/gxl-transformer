import os
import random
import torch
from torch.utils.data import Dataset, DataLoader

# 设置随机种子
random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)


class ShakespeareDataset(Dataset):
    def __init__(self, text, vocab, seq_len):
        self.vocab = vocab
        self.seq_len = seq_len
        self.data = self.text_to_indices(text)

    def text_to_indices(self, text):
        return [self.vocab[c] for c in text]

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y


def build_vocab(text):
    chars = sorted(list(set(text)))
    vocab = {c: i for i, c in enumerate(chars)}
    vocab_size = len(vocab)
    return vocab, vocab_size


def split_train_test(text, train_ratio=0.9):
    split_idx = int(len(text) * train_ratio)
    train_text = text[:split_idx]
    test_text = text[split_idx:]
    return train_text, test_text


def get_dataloaders(data_path, seq_len=64, batch_size=32):
    with open(data_path, 'r', encoding='utf-8') as f:
        text = f.read()

    train_text, test_text = split_train_test(text)
    vocab, vocab_size = build_vocab(text)

    train_dataset = ShakespeareDataset(train_text, vocab, seq_len)
    test_dataset = ShakespeareDataset(test_text, vocab, seq_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    return train_loader, test_loader, vocab, vocab_size