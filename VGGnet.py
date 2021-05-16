import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import math


device = torch.device("cuda:9")
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


class VGGnet(nn.Module):
    def __init__(self):
        super(VGGnet, self).__init__()
        self.relu = nn.ReLU()

        self.conv1a = nn.Conv2d(3, 8, 3, padding=1)
        self.conv1b = nn.Conv2d(8, 8, 3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2a = nn.Conv2d(8, 16, 3, padding=1)
        self.conv2b = nn.Conv2d(16, 16, 3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3a = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3b = nn.Conv2d(32, 32, 3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4a = nn.Conv2d(32, 64, 3, padding=1)
        self.conv4b = nn.Conv2d(64, 64, 3, padding=1)
        self.conv4c = nn.Conv2d(64, 64, 3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 10)
        self.fc3 = nn.Linear(10, 10)
        # initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()

    def forward(self, x):
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool1(x)

        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool2(x)

        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool3(x)

        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))
        x = self.relu(self.conv4c(x))
        x = self.pool4(x)

        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        # x = self.dropout(x)
        x = self.fc3(x)
        return x


VGGnet = VGGnet()
VGGnet.to(device)
VGGnet.load_state_dict(torch.load("./VGGnet.pth"))

criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(VGGnet.parameters(), lr=1e-4, momentum=0.9)
# lr=0.001, weight_decay=1e-5)
optimizer = torch.optim.Adam(VGGnet.parameters(), lr=1e-4, weight_decay=1e-5)

for epoch in range(5):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = VGGnet(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
#            torch.save(net.state_dict(), './cifar_net.pth')
#            print("saved")

print('Finished Training')
torch.save(VGGnet.state_dict(), "./VGGnet.pth")


correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        inputs, labels = data[0].to(device), data[1].to(device)
        outputs = VGGnet(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
