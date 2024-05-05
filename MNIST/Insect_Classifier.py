import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# 自定义数据集
class InsectDataset(Dataset):
    def __init__(self, file):
        self.X = []
        self.y = []
        with open(file, 'r') as f:
            for line in f:
                x, y, label = map(float, line.split())
                self.X.append([x, y])
                self.y.append(label)

        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# 定义模型
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(2, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 3)
        )

    def forward(self, input):
        output = self.model(input)
        return output

# 数据集路径
path = './insects/'

# 创建数据集
trainData = InsectDataset(path + 'insects-2-training.txt')
testData = InsectDataset(path + 'insects-2-testing.txt')

# 创建DataLoader
BATCH_SIZE = 10
trainDataLoader = DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
testDataLoader = DataLoader(dataset=testData, batch_size=BATCH_SIZE)

# 创建模型
net = Net()
print(net)

# 定义损失函数和优化器
lossF = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters())

# 训练模型
EPOCHS = 100
history = {'Test Loss': [], 'Test Accuracy': []}
for epoch in range(1, EPOCHS + 1):
    processBar = tqdm(trainDataLoader, unit='step')
    for step, (trainData, labels) in enumerate(processBar):
        net.zero_grad()
        outputs = net(trainData)
        loss = lossF(outputs, labels)
        predictions = torch.argmax(outputs, dim=1)
        accuracy = torch.sum(predictions == labels) / labels.shape[0]
        loss.backward()
        optimizer.step()
        processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f" %
                                   (epoch, EPOCHS, loss.item(), accuracy.item()))

        if step == len(processBar) - 1:
            correct, totalLoss = 0, 0
            for testData, labels in testDataLoader:
                outputs = net(testData)
                loss = lossF(outputs, labels)
                predictions = torch.argmax(outputs, dim=1)
                totalLoss += loss
                correct += torch.sum(predictions == labels)
            testAccuracy = correct / (BATCH_SIZE * len(testDataLoader))
            testLoss = totalLoss / len(testDataLoader)
            history['Test Loss'].append(testLoss.item())
            history['Test Accuracy'].append(testAccuracy.item())
            processBar.set_description("[%d/%d] Loss: %.4f, Acc: %.4f, Test Loss: %.4f, Test Acc: %.4f" %
                                       (epoch, EPOCHS, loss.item(), accuracy.item(), testLoss.item(),
                                        testAccuracy.item()))
    processBar.close()

# 可视化
plt.plot(history['Test Loss'], label='Test Loss')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

plt.plot(history['Test Accuracy'], color='red', label='Test Accuracy')
plt.legend(loc='best')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# 保存模型
torch.save(net, './model.pth')