'''
5.1.5.4 RNN - PyTorch (sin波)
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch
import torch.nn as nn
import torch.optim as optimizers
from callbacks import EarlyStopping


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, sequence):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.rnn_out = nn.Linear(hidden_size, output_size)
        self.fc = nn.Linear(sequence, output_size)


    def forward(self, x):
        h_out, h = self.rnn(x)
        out = self.rnn_out(h_out)
        y = self.fc(torch.permute(out, (0,2,1)))
        return y


if __name__ == '__main__':
    np.random.seed(123)
    torch.manual_seed(123)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''
    1. データの準備
    '''
    def sin(x, T=100):
        return np.sin(2.0 * np.pi * x / T)

    def toy_problem(T=100, ampl=0.05):
        x = np.arange(0, 2*T + 1)
        noise = ampl * np.random.uniform(low=-1.0, high=1.0,
                                         size=len(x))
        return sin(x) + noise

    T = 500
    f = toy_problem(T).astype(np.float32)
    length_of_sequences = len(f)
    step = 1
    maxlen = 25

    x = []
    t = []

    for i in range(length_of_sequences - maxlen - (step-1)):
        x.append(f[i:i+maxlen])
        t.append(f[i+maxlen+step-1])

    x = np.array(x).reshape(-1, maxlen, 1)
    t = np.array(t).reshape(-1, 1)

    x_train, x_val, t_train, t_val = \
        train_test_split(x, t, test_size=0.2, shuffle=False)

    '''
    2. モデルの構築
    '''
    input_size = 1  # 入力数
    hidden_size = 50  # 隠れ層の次元
    output_size = 1  # 出力数
    sequence = 25  # シーケンス数

    model = RNN(input_size, hidden_size, output_size, sequence).to(device)

    '''
    3. モデルの学習
    '''
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optimizers.Adam(model.parameters(),
                                lr=0.001,
                                betas=(0.9, 0.999), amsgrad=True)

    def compute_loss(t, y):
        return criterion(y, t)

    def train_step(x, t):
        x = torch.Tensor(x).to(device)
        t = torch.Tensor(t).to(device)
        model.train()
        preds = model(x)
        loss = compute_loss(t, preds)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss, preds

    def val_step(x, t):
        x = torch.Tensor(x).to(device)
        t = torch.Tensor(t).to(device)
        model.eval()
        preds = model(x)
        loss = criterion(preds, t)

        return loss, preds

    epochs = 1000
    batch_size = 100
    n_batches_train = x_train.shape[0] // batch_size + 1
    n_batches_val = x_val.shape[0] // batch_size + 1
    hist = {'loss': [], 'val_loss': []}
    es = EarlyStopping(patience=10, verbose=1)

    for epoch in range(epochs):
        train_loss = 0.
        val_loss = 0.
        x_, t_ = shuffle(x_train, t_train)

        for batch in range(n_batches_train):
            start = batch * batch_size
            end = start + batch_size
            loss, _ = train_step(x_[start:end], t_[start:end])
            train_loss += loss.item()

        for batch in range(n_batches_val):
            start = batch * batch_size
            end = start + batch_size
            loss, _ = val_step(x_val[start:end], t_val[start:end])
            val_loss += loss.item()

        train_loss /= n_batches_train
        val_loss /= n_batches_val

        hist['loss'].append(train_loss)
        hist['val_loss'].append(val_loss)

        print('epoch: {}, loss: {:.3}, val_loss: {:.3f}'.format(
            epoch+1,
            train_loss,
            val_loss
        ))

        if es(val_loss):
            break

    # モデルの保存
    model_path = 'model/sin_rnn_model.pth'
    torch.save(model.state_dict(), model_path)

    '''
    4. モデルの評価
    '''
    model.eval()

    # sin波の予測
    sin = toy_problem(T, ampl=0.)
    gen = [None for i in range(maxlen)]


    z = x[:1]

    for i in range(length_of_sequences - maxlen):
        z_ = torch.Tensor(z[-1:]).to(device)
        preds = model(z_).data.cpu().numpy()
        z = np.append(z, preds)[1:]
        z = z.reshape(-1, maxlen, 1)
        gen.append(preds[0, 0])
    print(type(gen))
    print(gen)
    # 予測値を可視化
    fig = plt.figure()
    plt.rc('font', family='serif')
    plt.xlim([0, 2*T])
    plt.ylim([-1.5, 1.5])
    plt.plot(range(len(f)), sin,
             color='gray',
             linestyle='--', linewidth=0.5)
    plt.plot(range(len(f)), gen,
             color='black', linewidth=1,
             marker='o', markersize=1, markerfacecolor='black',
             markeredgecolor='black')
    # plt.savefig('output.jpg')
    plt.show()
