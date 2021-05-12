import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


device = torch.device("cuda:5")
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4
trainset = torchvision.datasets.CIFAR10(root='/data3/kaleb.dickerson2001/CIFAR_Data',
                                        train=True,
                                        download=True,
                                        transform=transform)
trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)

testset = torchvision.datasets.CIFAR10(root='/data3/kaleb.dickerson2001/CIFAR_Data',
                                       train=False,
                                       download=True,
                                       transform=transform)
testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=batch_size,
                                         shuffle=False,
                                         num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')


class googLeNet(nn.Module):
    def __init__(self):
        super(googLeNet, self).__init__()
        self.conv1 = conv_block(in_channels=3, out_channels=32, kernel_size=(5, 5),
                                stride=(1, 1), padding=(2, 2))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)
        self.conv2 = conv_block(
            in_channels=32, out_channels=96, kernel_size=(3, 3), stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.inception3a = inception_block(
            in_channels=96, out_1x1=48, red_3x3=32, out_3x3=96, red_5x5=8, out_5x5=24, out_1x1maxpool=24)
        self.inception3b = inception_block(
            in_channels=192, out_1x1=56, red_3x3=86, out_3x3=154, red_5x5=16, out_5x5=48, out_1x1maxpool=32)

        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.inception4a = inception_block(
            in_channels=290, out_1x1=192, red_3x3=96, out_3x3=208, red_5x5=16, out_5x5=48, out_1x1maxpool=64)
        self.inception4b = inception_block(
            in_channels=512, out_1x1=160, red_3x3=112, out_3x3=224, red_5x5=24, out_5x5=64, out_1x1maxpool=64)
        self.inception4c = inception_block(
            in_channels=512, out_1x1=128, red_3x3=128, out_3x3=256, red_5x5=24, out_5x5=64, out_1x1maxpool=64)
        self.inception4d = inception_block(
            in_channels=512, out_1x1=112, red_3x3=144, out_3x3=288, red_5x5=32, out_5x5=64, out_1x1maxpool=64)
        self.inception4e = inception_block(
            in_channels=528, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1maxpool=128)

        self.maxpool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

        self.inception5a = inception_block(
            in_channels=290, out_1x1=256, red_3x3=160, out_3x3=320, red_5x5=32, out_5x5=128, out_1x1maxpool=128)
        self.inception5b = inception_block(
            in_channels=832, out_1x1=384, red_3x3=192, out_3x3=384, red_5x5=48, out_5x5=128, out_1x1maxpool=128)

        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(9216, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # x = self.inception4a(x)
        # x = self.inception4b(x)
        # x = self.inception4c(x)
        # x = self.inception4d(x)
        # x = self.inception4e(x)
        # x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        # x = self.dropout(x)
        x = self.fc1(x)
        return x


class inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1maxpool):
        super(inception_block, self).__init__()
        self.branch1 = conv_block(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            conv_block(in_channels, red_3x3, kernel_size=1),
            conv_block(red_3x3, out_3x3, kernel_size=3, padding=1)
        )
        self.branch3 = nn.Sequential(
            conv_block(in_channels, red_5x5, kernel_size=1),
            conv_block(red_5x5, out_5x5, kernel_size=5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, out_1x1maxpool, kernel_size=1)
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.relu(self.batchnorm(self.conv(x)))


googLeNet = googLeNet()
googLeNet.to(device)
googLeNet.load_state_dict(torch.load("./customGoogLeNet.pth"))

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(googLeNet.parameters(), lr=1e-6, weight_decay=1e-5)

# for epoch in range(2):  # loop over the dataset multiple times

#     running_loss = 0.0
#     for i, data in enumerate(trainloader, 0):
#         # get the inputs; data is a list of [inputs, labels]
#         inputs, labels = data[0].to(device), data[1].to(device)

#         # zero the parameter gradients
#         optimizer.zero_grad()

#         # forward + backward + optimize
#         outputs = googLeNet(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()

#         # print statistics
#         running_loss += loss.item()
#         if i % 2000 == 1999:    # print every 2000 mini-batches
#             print('[%d, %5d] loss: %.3f' %
#                   (epoch + 1, i + 1, running_loss / 2000))
#             running_loss = 0.0
# #            torch.save(net.state_dict(), './cifar_net.pth')
# #            print("saved")

# print('Finished Training')
# torch.save(googLeNet.state_dict(), './customGoogLeNet.pth')


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = googLeNet(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
