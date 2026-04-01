"""Recurrent networks for text classification using PyTorch."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.datasets import fetch_20newsgroups
from collections import Counter
import re


class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=200):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.tokenize(self.texts[idx])[:self.max_len]
        indices = [self.vocab.get(token, 0) for token in tokens]

        # Pad sequences
        if len(indices) < self.max_len:
            indices += [0] * (self.max_len - len(indices))

        return torch.tensor(indices), torch.tensor(self.labels[idx])

    @staticmethod
    def tokenize(text):
        text = text.lower()
        return re.findall(r'\b[a-z]+\b', text)


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        x = torch.relu(self.fc1(hidden))
        x = self.dropout(x)
        return self.fc2(x)


def load_data():
    print("Loading 20newsgroups dataset...")
    categories = ['alt.atheism', 'soc.religion.christian']
    newsgroups = fetch_20newsgroups(subset='all', categories=categories, shuffle=True, random_state=42)

    texts = newsgroups.data
    labels = newsgroups.target

    return texts, labels


def build_vocabulary(texts, max_vocab=5000):
    print("Building vocabulary...")
    all_tokens = []
    for text in texts:
        all_tokens.extend(TextDataset.tokenize(text))

    counter = Counter(all_tokens)
    vocab = {'<pad>': 0, '<unk>': 1}
    for word, _ in counter.most_common(max_vocab - 2):
        vocab[word] = len(vocab)

    return vocab


def create_dataloaders(texts, labels, vocab, batch_size=32):
    print("Creating dataloaders...")
    dataset = TextDataset(texts, labels, vocab)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader


def train_model(model, train_loader, device, epochs=3):
    print("Training model...")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss, correct = 0, 0

        for texts, labels in train_loader:
            texts, labels = texts.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()

        accuracy = correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}, Accuracy: {accuracy:.4f}")


def evaluate_model(model, test_loader, device):
    print("\nEvaluating model...")
    model.eval()
    correct = 0

    with torch.no_grad():
        for texts, labels in test_loader:
            texts, labels = texts.to(device), labels.to(device)
            outputs = model(texts)
            correct += (outputs.argmax(1) == labels).sum().item()

    accuracy = correct / len(test_loader.dataset)
    print(f"Test Accuracy: {accuracy:.4f}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load and prepare data
    texts, labels = load_data()
    vocab = build_vocabulary(texts)
    train_loader, test_loader = create_dataloaders(texts, labels, vocab)

    # Create and train model
    model = LSTMClassifier(vocab_size=len(vocab), num_classes=2).to(device)
    train_model(model, train_loader, device, epochs=3)
    evaluate_model(model, test_loader, device)


if __name__ == "__main__":
    main()