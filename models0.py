import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import torch.cuda

def gpu(x):
    return x.cuda() if torch.cuda.is_available() else x

## define the convolutional neural network architecture : intial version 0
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
           
        ## Define all the layers of this CNN, maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## The last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5) ## output size = (W-F+2P)/S +1 = (224-5+2)/2 +1 = 112
        
        # maxpool layer: pool with kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2, 2)
        
        # fully-connected layer 32*4 input size to account for the downsampled image size after pooling
        # num_classes outputs (for n_classes of image data): 136 output
        self.fc1 = nn.Linear(32*4, 136)
        

        # define the feedforward behavior
    def forward(self, x):
        ## Define the feedforward behavior of this model
        ## x is the input image and a pool/conv step:
        # one conv/relu + pool layers
        x = self.pool(F.relu(self.conv1(x)))
        
        # prep for linear layer by flattening the feature maps into feature vectors
        x = x.view(x.size(0), -1)
        # linear layer
        x = F.relu(self.fc1(x))
        # final output
        # droout: x= self.fc1_drop(x)
        # a modified x, having gone through all the layers of the model, should be returned
        return x
