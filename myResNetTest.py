import torch
from torch.nn.modules.pooling import MaxPool1d, MaxPool2d
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F


device = torch.device("cuda:8")
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


class ResNet(nn.Module):
    def __init__(self, numClasses=10):
        super(ResNet, self).__init__()
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 64, 5, 2, 2)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv1a = nn.Conv2d(64, 64, 3, 1, 1)
        self.conv2a = nn.Conv2d(64, 64, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(64)

        self.res_1x1b = nn.Conv2d(64, 128, 1, 2, 0)
        self.conv1b = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv2b = nn.Conv2d(128, 128, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn_res3 = nn.BatchNorm2d(128)

        self.res_1x1c = nn.Conv2d(128, 256, 1, 2, 0)
        self.conv1c = nn.Conv2d(128, 256, 3, 2, 1)
        self.conv2c = nn.Conv2d(256, 256, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn_res4 = nn.BatchNorm2d(256)

        self.res_1x1d = nn.Conv2d(256, 512, 1, 2, 0)
        self.conv1d = nn.Conv2d(256, 512, 3, 2, 1)
        self.conv2d = nn.Conv2d(512, 512, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(512)
        self.bn_res5 = nn.BatchNorm2d(512)

        self.avgpool = nn.AvgPool2d(3, 2, 1)
        self.fc1 = nn.Linear(2048, numClasses)

    def forward(self, x):
        out = x
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv1a(out)
        out = self.conv2a(out)
        out = self.relu(self.bn2(out))

        res = out
        out = self.conv1b(out)
        out = self.conv2b(out)
        out = self.bn3(out)
        out += self.bn_res3(self.res_1x1b(res))
        out = self.relu((out))

        res = out
        out = self.conv1c(out)
        out = self.conv2c(out)
        out = self.bn4(out)
        out += self.bn_res4(self.res_1x1c(res))
        out = self.relu((out))

        res = out
        out = self.conv1d(out)
        out = self.conv2d(out)
        out = self.bn5(out)
        out += self.bn_res5(self.res_1x1d(res))
        out = self.relu((out))

        out = out.reshape(x.shape[0], -1)
        out = self.fc1(out)
        return out


resnet = ResNet().to(device)
# resnet.load_state_dict(torch.load("./cifar_resnetTEST.pth"))

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(resnet.parameters(), lr=0.01, weight_decay=1e-5)

for epoch in range(8):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = resnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
#            torch.save(resnet.state_dict(), './cifar_resnetTEST.pth')
#            print("saved")

print('Finished Training')
torch.save(resnet.state_dict(), './cifar_resnetTEST.pth')


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = resnet(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the resnetwork on the 10000 test images: %d %%' % (
    100 * correct / total))
