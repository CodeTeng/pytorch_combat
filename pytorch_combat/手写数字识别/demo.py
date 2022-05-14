#! /usr/bin/env python3
# -*- coding: utf-8 -*-

# @Author: 狂小腾
# @Date: 2022/5/14 15:51

import torch
import torchvision.transforms
import torch.nn.functional as F
import numpy as np
import cv2
from torch import nn
from torch.utils.data import DataLoader

# 定义超参数
batch_size = 64  # 每次批处理的数据
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
epoch = 50  # 训练数据集的轮数

# 转换层 对图像做处理
transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.1307,), (0.3081,))  # 正则化 降低模型复杂度
])

# 获取数据集
train_data = torchvision.datasets.MNIST("./datasets", train=True, download=True, transform=transform)
test_data = torchvision.datasets.MNIST("./datasets", train=False, download=True, transform=transform)

train_data_size = len(train_data)
test_data_size = len(test_data)
print(f"训练数据集长度为：{train_data_size}")
print(f"测试数据集长度为：{test_data_size}")

# 加载数据集
train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True)

# 显示MNIST中的图片
with open('./datasets/MNIST/raw/train-images-idx3-ubyte', 'rb') as f:
    file = f.read()

image1 = [int(str(item).encode('ascii'), 16) for item in file[16: 16 + 784]]
image1_np = np.array(image1, dtype=np.uint8).reshape(28, 28, 1)  # 通道是1 灰度
print(image1_np.shape)
cv2.imwrite("digit.jpg", image1_np)


# 构建网络模型
class Model(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        # 1:灰度图片的通道 10：输出通道 5：卷积核大小 1：步长
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(in_features=20 * 10 * 10, out_features=500)  # 20*10*10:输入通道 500：输出通道
        self.fc2 = nn.Linear(in_features=500, out_features=10)  # 500:输入通道  10：输出通道 0-9

    def forward(self, x):
        input_size = x.size(0)  # batch_size   x.size()--> batch_size*channel*height*weight
        x = self.conv1(x)  # 输入：batch*1*28*28   输出：batch*10*24*24   (28 - 5 + 1)
        x = F.relu(x)  # 保持shape不变  输出：batch*10*24*24
        x = F.max_pool2d(x, 2, 2)  # 输入：batch*10*24*24  输出：batch*10*12*12   最大池化图片减半

        x = self.conv2(x)  # 输入：batch*10*12*12  输出：batch*20*10*10   (12 - 3 + 1)
        x = F.relu(x)  # 保持shape不变 输出：batch*20*10*10

        x = x.view(input_size, -1)  # 拉平，将4维降低为1维 -1：自动计算维度，20*10*10=2000

        x = self.fc1(x)  # 输入：batch*2000    输出：batch*500
        x = F.relu(x)  # 保持shape不变

        x = self.fc2(x)  # 输入：batch*500     输出：batch*10

        output = F.log_softmax(input=x, dim=1)  # 计算分类后，每个数字的概率值

        return output


# 创建网络模型
model = Model().to(device)

# 损失函数
loss_fn = nn.CrossEntropyLoss().to(device)

# 优化器
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)


# 定义训练方法
def train_model(model, device, train_loader, optimizer, loss_fn, epoch):
    # 模型训练
    model.train()
    for batch_index, (imgs, targets) in enumerate(train_loader):
        imgs, targets = imgs.to(device), targets.to(device)
        # 梯度初始化为0
        optimizer.zero_grad()
        # 训练后的结果
        outputs = model(imgs)
        # 计算损失
        loss = loss_fn(outputs, targets)
        # 反向传播
        loss.backward()
        # 参数优化
        optimizer.step()

        if batch_index % 3000 == 0:
            print(f"训练次数：{epoch}，损失率：{loss.item():.6f}")


# 定义测试方法
def test_model(model, device, test_loader, loss_fn):
    # 模型验证
    model.eval()
    # 精确率
    accuracy = 0.0
    # 测试损失
    test_loss = 0.0
    with torch.no_grad():  # 不会计算梯度，也不会进行反向传播
        for imgs, targets in test_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            # 测试数据
            outputs = model(imgs)
            # 计算测试损失
            loss = loss_fn(outputs, targets).item()
            test_loss += loss
            # 计算精确率
            accuracy += (outputs.argmax(1) == targets).sum().item()
        test_loss /= test_data_size
        print(f"测试数据---平均损失率：{test_loss:.4f}，精确率：{(100 * accuracy / test_data_size):.3f}")


if __name__ == '__main__':
    for i in range(1, epoch + 1):
        train_model(model, device, train_loader, optimizer, loss_fn, i)
        test_model(model, device, test_loader, loss_fn)
