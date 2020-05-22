import torch
import torch.nn as nn
from tensorboardX import SummaryWriter


class TestModel(nn.Module):
    def __init__(self, size=4):
        super(TestModel, self).__init__()
        self.conv = nn.Conv2d(3, 16, 3, 2, 1)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(True)
        self.linear = nn.Linear(16 * size * size, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = x.flatten()
        x = self.linear(x)
        x = nn.Sigmoid()(x)
        return x


if __name__ == '__main__':
    x = torch.rand(1, 3, 64, 64)
    print(x)
    model = TestModel(size=32)
    out = model(x)
    print(out)
    criterion = nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr=1e-4)
    writter = SummaryWriter(log_dir='data/log', comment='test')
    for i in range(120):
        out = model(x)
        loss = criterion(out, torch.tensor([0.1]))
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(loss)
        writter.add_scalar('loss', loss, i)
    writter.close()
