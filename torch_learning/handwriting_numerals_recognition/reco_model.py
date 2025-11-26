import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as op
from Net import *

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,))])
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
train_loader = DataLoader(trainset, batch_size=64, shuffle=True)
test_loader = DataLoader(testset, batch_size=64, shuffle=False)

model = Net()
learning_rate = 1e-3
optimizer = op.Adam(model.parameters(), lr=learning_rate)
lossF = nn.CrossEntropyLoss()

writer = SummaryWriter("./logs")
total_train_steps = 0
total_test_steps = 0

# 训练代码
model.train()
for epoch in range(10):
    print(f'-----第{epoch+1}轮训练-----')
    running_loss = 0.0
    for data in train_loader:
        imgs,targets = data
        outputs = model(imgs)
        loss = lossF(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        total_train_steps += 1
        if total_train_steps % 10 == 0:
            print(f"已训练{total_train_steps}步,总loss：{running_loss}")
            writer.add_scalar("train_loss", loss, total_train_steps)
            writer.flush()

    print(f"第{epoch+1}轮训练，训练总loss:{running_loss}")

# 测试代码
    total_test_loss = 0
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            loss = lossF(outputs, labels)
            total_test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'第{epoch+1}轮训练后的测试准确率: {100 * correct / total:.2f}%')
    total_test_steps += 1
    writer.add_scalar("test_loss", total_test_loss, total_test_steps)
    writer.add_scalar("accuracy", correct / total, total_test_steps)
    writer.flush()

torch.save(model.state_dict(), './final_model1.pth')
print("训练完毕，模型已保存")
writer.close()


