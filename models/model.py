import os 
import torch
import numpy as np 
import cv2
import torchvision
from torchvision import *
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import *
import torch.optim as optim
import time
import numpy as np
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)
class Handwritten_Recognition:
    def __init__(self):
        self.model = MyModel()
        self.model.load_state_dict(torch.load(r"/home/nguyen.viet.hoai/Documents/Hoai/deploy web/checkpoint/model_handwritten_recognition.pth"))
        self.trans  = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])
    def predict(self, X):
        return self.model(X)
    def predict_image(self, image_path):
        img = cv2.imread(image_path)
        img =cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img/255.
        img = np.reshape(img, (1, 28, 28, 1))
        img = self.trans(img)
        return self.predict(img)