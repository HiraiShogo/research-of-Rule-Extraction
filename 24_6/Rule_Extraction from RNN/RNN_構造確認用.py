import torch
import torch.nn as nn

# シンプルなRNNモデルの定義
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sequence):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.rnn_out = nn.Linear(hidden_size, output_size)
        self.fc = nn.Linear(sequence, output_size)

    def forward(self, x):
        h_out, h = self.rnn(x)
        out = self.rnn_out(h_out)
        y = self.fc(torch.permute(out, (0,2,1)))
        return h_out, h, out, y

    def init_hidden(self, batch_size):
        return torch.zeros(1, batch_size, self.hidden_size)

# モデルの初期化
input_size = 1  # 入力の次元
hidden_size = 10  # 隠れ状態の次元
output_size = 1  # 出力の次元
batch_size = 1  # バッチサイズ
sequence = 3  # シーケンス長

model = SimpleRNN(input_size, hidden_size, output_size, sequence)

# ダミー入力データの生成
input_data = torch.randn(batch_size, sequence, input_size)  # (バッチサイズ, シーケンス長, 入力の次元)
#hidden_state = model.init_hidden(batch_size)

# フォワードパスを実行して全ての中間状態を取得
h_out, hidden_state, out, y = model(input_data)

# 全ての中間状態の表示
print("入力 : ", input_data.size())
print("h_outのサイズ:", h_out.size())
print("hidden_stateのサイズ:", hidden_state.size())
print("RNN出力 : ", out.size())
print("全結合層出力 : ", y.size())

