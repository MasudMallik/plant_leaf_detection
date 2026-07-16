from torch import nn
from torchvision import transforms
import torch

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(in_channels=3,out_channels=12,kernel_size=3)
        self.conv2=nn.Conv2d(in_channels=12,out_channels=24,kernel_size=3)
        self.conv3=nn.Conv2d(in_channels=24,out_channels=28,kernel_size=3)
        self.pool=nn.MaxPool2d(kernel_size=3,stride=3)
        self.relu=nn.ReLU(inplace=True)

        self.fc1=nn.Linear(in_features=28*7*7,out_features=1513)
        self.fc2=nn.Linear(in_features=1513,out_features=800)
        self.fc3=nn.Linear(in_features=800,out_features=400)
        self.fc4=nn.Linear(in_features=400,out_features=12)
    def forward(self,x):
        x=self.pool(self.relu(self.conv1(x)))
        x=self.pool(self.relu(self.conv2(x)))
        x=self.pool(self.relu(self.conv3(x)))
        x=torch.flatten(x,1)
        return (self.fc4(self.relu(self.fc3(self.relu(self.fc2(self.relu(self.fc1(x))))))))
        
testing_trans=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])