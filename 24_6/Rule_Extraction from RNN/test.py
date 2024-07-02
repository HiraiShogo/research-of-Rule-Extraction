import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import sys

# 定数の設定
seq_length = 20
n_steps = 1000
input_size = 1
hidden_size = 50
num_layers = 2
output_size = 1
num_epochs = 100
learning_rate = 0.01

# SIN波データの生成
time_steps = np.linspace(0, np.pi * 2 * n_steps / 100, n_steps)
print(time_steps)
sys.exit()
data = np.sin(time_steps)


# データの前処理
def create_inout_sequences(data, seq_length):
    inout_seq = []
    for i in range(len(data) - seq_length):
        train_seq = data[i:i + seq_length]
        train_label = data[i + seq_length]
        inout_seq.append((train_seq, train_label))
    return inout_seq


train_sequences = create_inout_sequences(data, seq_length)


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out


# モデル、ロス関数、最適化関数の設定
model = RNN(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# トレーニング
for epoch in range(num_epochs):
    for seq, labels in train_sequences:
        seq = torch.FloatTensor(seq).view(-1, seq_length, input_size)
        labels = torch.FloatTensor([labels])

        outputs = model(seq)
        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 予測
test_inputs = data[:seq_length].tolist()
model.eval()
for _ in range(n_steps - seq_length):
    seq = torch.FloatTensor(test_inputs[-seq_length:]).view(-1, seq_length, input_size)
    with torch.no_grad():
        test_inputs.append(model(seq).item())

# 結果のプロット
plt.plot(time_steps, data, label='Real Data')
plt.plot(time_steps[seq_length:], test_inputs[seq_length:], label='Predicted Data')
plt.legend()
plt.show()
