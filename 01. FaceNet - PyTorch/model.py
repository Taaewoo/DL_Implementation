import torch.nn as nn
import torch.nn.functional as F
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from PIL import Image
from matplotlib import pyplot
from torchvision.models import resnet50
from torchsummary import summary

class Flatten(nn.Module):
    def forward(self,x):
        return x.view(x.size(0),-1)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        # input depth, output depth, image size
        self.conv1 = nn.Conv2d(3,64,7,stride=2,padding=3)
        # kernel_size, stride
        self.pool1 = nn.MaxPool2d(3,2,padding=1)
        # hitten size ( channel )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv2a = nn.Conv2d(64,64,1,stride=1)
        self.conv2 = nn.Conv2d(64,192,3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(192)
        self.pool2 = nn.MaxPool2d(3,2,padding=1)
        self.conv3a = nn.Conv2d(192,192,1,stride=1)
        self.conv3 = nn.Conv2d(192,384,3,stride=1,padding=1)
        self.pool3 = nn.MaxPool2d(3,2,padding=1)
        self.conv4a = nn.Conv2d(384,384,1,stride=1)
        self.conv4 = nn.Conv2d(384,256,3,stride=1,padding=1)
        self.conv5a = nn.Conv2d(256,256,1,stride=1)
        self.conv5 = nn.Conv2d(256,256,3,stride=1,padding=1)
        self.conv6a = nn.Conv2d(256,256,1,stride=1)
        self.conv6 = nn.Conv2d(256,256,3,stride=1,padding=1)
        self.pool4 = nn.MaxPool2d(3,2,padding=1)
        self.fc1 = nn.Linear(256*7*7, 32*128)
        self.fc2 = nn.Linear(32*128, 32*128)
        self.fc7128 = nn.Linear(32*128, 128)


        self.cnn = nn.Sequential(
                self.conv1,
                self.pool1,
                self.bn1,
                self.relu,
                self.conv2a,
                self.conv2,
                self.bn2,
                self.relu,
                self.pool2,
                self.conv3a,
                self.conv3,
                self.relu,
                self.pool3,
                self.conv4a,
                self.conv4,
                self.relu,
                self.conv5a,
                self.conv5,
                self.relu,
                self.conv6a,
                self.conv6,
                self.relu,
                self.pool4,
                Flatten(),
                self.fc1,
                self.relu,
                self.fc2,
                self.relu,
                self.fc7128,
        )

        summary(self.cnn, (3,220,220) )

    def forward(self, x):
        x = self.cnn(x)
        return x
