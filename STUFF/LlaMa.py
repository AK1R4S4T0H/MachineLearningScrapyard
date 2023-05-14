import os
import torch
import torch.nn as nn
import pickle
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class LocallyLinearAttention(nn.Module):
    def __init__(self, input_size, neighborhood_size):
        super().__init__()
        self.input_size = input_size
        self.neighborhood_size = neighborhood_size
        self.query = nn.Linear(input_size, input_size, bias=False)
        self.key = nn.Linear(input_size, input_size, bias=False)
        self.value = nn.Linear(input_size, input_size, bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        batch_size, seq_len, input_size = x.size()
        queries = self.query(x)
        keys = self.key(x)
        values = self.value(x)
        queries = queries.view(batch_size, seq_len, self.neighborhood_size, input_size // self.neighborhood_size)
        keys = keys.view(batch_size, seq_len, self.neighborhood_size, input_size // self.neighborhood_size)
        values = values.view(batch_size, seq_len, self.neighborhood_size, input_size // self.neighborhood_size)
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scores = scores / (input_size // self.neighborhood_size) ** 0.5
        attention_weights = self.softmax(scores)
        context = torch.matmul(attention_weights, values)
        context = context.view(batch_size, seq_len, input_size)
        return context


class LLAMALanguageModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, neighborhood_size, seq_len):
        super().__init__()
        self.tokenizer = lambda s: s.split()  # default whitespace tokenizer
        self.preprocessor = lambda x: x  # default no-op preprocessor
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            LocallyLinearAttention(hidden_size, neighborhood_size) for _ in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.seq_len = seq_len

    def forward(self, x):
        batch_size, seq_len = x.size()
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(batch_size * seq_len, -1)
        x = self.fc(x)
        x = x.reshape(batch_size, seq_len, -1)
        return x

class LanguageModelDataset(Dataset):
    def __init__(self, data, seq_len):
        self.seq_len = seq_len
        self.data = data
        self.vocab = set(data)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, index):
        inputs = self.data[index:index + self.seq_len]
        targets = self.data[index + 1:index + self.seq_len + 1]
        return torch.tensor(inputs), torch.tensor(targets)

def load_data_path(model, tokenizer, preprocessor, data_path):
    files = os.listdir(data_path)
    data = []
    for file in files:
        with open(os.path.join(data_path, file), 'r', encoding='utf-8') as f:
            data.extend(tokenizer(preprocessor(f.read())))
    
    token_to_id = {token: i for i, token in enumerate(set(data))}
    id_to_token = {i: token for token, i in token_to_id.items()}
    data = [token_to_id[token] for token in data]
    
    dataset = LanguageModelDataset(data, model.seq_len)
    return dataset, id_to_token


def train(model, tokenizer, preprocessor, data_path, epochs, batch_size, lr, device):
    dataset, id_to_token = load_data_path(model, tokenizer, preprocessor, data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    model.to(device)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 100 == 99:
                print(f"Epoch {epoch + 1}, Batch {i + 1}, Loss {running_loss / 100}")
                running_loss = 0.0
    
    model_path = "model.pt"
    torch.save(model.state_dict(), model_path)
    
    tokenizer_path = "tokenizer.pt"
    torch.save(tokenizer, tokenizer_path)
    
    id_to_token_path = "id_to_token.pkl"
    with open(id_to_token_path, "wb") as f:
        pickle.dump(id_to_token, f)
    
    return id_to_token
# train('model.pt', 'tokenizer.pt', '/content/sample_data/mnist_train_small.csv', epochs=100, batch_size=32, lr=0.1, device='cpu')