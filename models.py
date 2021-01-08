## define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):
    
    # parameters
    RANDOM_SEED = 42
    LEARNING_RATE = 0.001
    BATCH_SIZE = 32
    N_EPOCHS = 15

    IMG_SIZE = 32
    N_CLASSES = 10

    def __init__(self):
        super(Net, self).__init__()
        
        ## Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## 3. The last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        

        # conv1-1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5) # output size = (W-F)/S +1 = (224-5)/1 +1 = 220
        self.pool1 = nn.MaxPool2d(2, 2)# pool with kernel_size=2, stride=2, output = 220/2=110
        self.conv2 = nn.Conv2d(32, 64, 3) # output size = (W-F)/S +1 = (110-3)/1 + 1 = 108
        self.pool2 = nn.MaxPool2d(2, 2) # pool with kernel_size=2, stride=2, output=108/2=54
        self.conv3 = nn.Conv2d(64, 128, 3) # output size = (W-F)/S +1 = (54-3)/1 + 1 = 52
        self.pool3 = nn.MaxPool2d(2, 2) # pool with kernel_size=2, stride=2, output=52/2=26
        self.conv4 = nn.Conv2d(128, 256, 3) # output size = (W-F)/S +1 = (26-3)/1 + 1 = 24
        self.pool4 = nn.MaxPool2d(2, 2) # pool with kernel_size=2, stride=2, output=24/2=12
        self.conv5 = nn.Conv2d(256, 512, 1) # output size = (W-F)/S +1 = (12-3)/1 + 1 = 10
        self.pool5 = nn.MaxPool2d(2, 2)
        # Fully-connected (linear) layers
        #self.fc1 = nn.Linear(52*52*128, 512)
        #self.fc2 = nn.Linear(512, 68*2)
        self.fc1 = nn.Linear(512*6*6, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 68*2)
        # Dropout
        self.dropout1 = nn.Dropout(p=0.25)
        self.dropout2 = nn.Dropout(p=0.25)
 
        

        
    def forward(self, x):
        ## the feedforward behavior of this model
        ## x is the input image and, 5 conv/relu + MaxPooling layers and before the output, two fully connected layers with two dropouts(to avoid overfitting) and the last layer is fully connected layer
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = self.pool4(F.relu(self.conv4(x)))
        x = self.pool4(F.relu(self.conv5(x)))
        
        # Prep for linear layer / Flatten
        x = x.view(x.size(0), -1)
        
        # linear layers with dropout in between
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        
        return x
