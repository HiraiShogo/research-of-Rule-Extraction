import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim

# RNNモデルの定義
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 初期隠れ状態の初期化
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        # RNNの出力
        out, _ = self.rnn(x, h0)
        # 最後のタイムステップの出力を使用して分類
        out = self.fc(out[:, -1, :])
        return out

# デバイスの設定
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ハイパーパラメータの設定
sequence_length = 28
input_size = 28
hidden_size = 12
num_layers = 1
num_classes = 10
batch_size = 100
num_epochs = 10

# MNISTデータセットのダウンロードおよびデータローダーの作成
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trans=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.5,),(0.5,))])

train_data=torchvision.datasets.MNIST(root="mnist_data",train=True,download=True,transform=trans)
train_loader=torch.utils.data.DataLoader(train_data,batch_size=100,shuffle=True,num_workers=0)

test_data=torchvision.datasets.MNIST(root="mnist_data",train=False,download=True,transform=trans)
test_loader=torch.utils.data.DataLoader(test_data,batch_size=100,shuffle=False,num_workers=0)

test_data_ext=test_data[0][0].view(1,1,28,28)
test_data_ext=test_data_ext.to(device)

# モデルの初期化
model = RNN(input_size, hidden_size, num_layers, num_classes).to(device)

# 損失関数とオプティマイザの設定
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# トレーニングループ
total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        # 入力をReshapeして、(batch_size, sequence_length, input_size)の形にする
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)

        # 順伝播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 逆伝播と最適化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_step}], Loss: {loss.item():.4f}')

# テストループ
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, sequence_length, input_size).to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print(f'Test Accuracy of the model on the 10000 test images: {100 * correct / total:.2f}%')

# モデルの保存
torch.save(model.state_dict(), 'rnn_model.ckpt')