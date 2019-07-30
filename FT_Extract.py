import torch
import torch.nn as nn
import torch.nn.functional as F



class FeatureExtraction(nn.Module):
    """
    This is a sequential model starting with two inception layers at the front, 
    followed by five convolutional layers of. 
    The output of the last layer is supposed to contain sufficient features to facilitate classification.
    The output of the last layer can be used as the input to a transposed CNN network (deconvolutional network) to
    reconstruct the input.
    """

    def __init__(self):
        super(FeatureExtraction, self).__init__()
        #
        # input size = (_, 3, 240, 11)
        self.inc11 = nn.Conv2d(  3,  10, kernel_size=(45, 1), stride=(1, 1), padding=(22, 0))
        self.inc12 = nn.Conv2d(  3,  10, kernel_size=( 1, 5), stride=(1, 1), padding=( 0, 2))
        self.inc13 = nn.Conv2d(  3,  10, kernel_size=( 5, 5), stride=(1, 1), padding=( 2, 2))
        #
        self.inc21 = nn.Conv2d( 30,  20, kernel_size=(25, 1), stride=(1, 1), padding=(12, 0))
        self.inc22 = nn.Conv2d( 30,  20, kernel_size=( 1, 3), stride=(1, 1), padding=( 0, 1))
        self.inc23 = nn.Conv2d( 30,  20, kernel_size=( 3, 3), stride=(1, 1), padding=( 1, 1))
        #
        '''
        # replace first two inception layers with two regular convolutional layers that produce the same number of channels 
        self.conv01= nn.Conv2d(  3,  30, kernel_size=( 5, 5), stride=(1, 1), padding=( 2, 2))
        self.conv02= nn.Conv2d( 30,  60, kernel_size=( 3, 3), stride=(1, 1), padding=( 1, 1))
        '''
        #
        # regular convolutional layers:
        self.conv1 = nn.Conv2d( 60, 100, kernel_size=( 3, 2), stride=(1, 1))
        #
        self.conv2 = nn.Conv2d(100, 120, kernel_size=( 3, 3), stride=(1, 1), padding=( 1, 1))
        #
        self.conv3 = nn.Conv2d(120, 140, kernel_size=( 3, 3), stride=(1, 1), padding=( 1, 1))
        #
        self.conv4 = nn.Conv2d(140, 160, kernel_size=( 3, 3), stride=(1, 1), padding=( 1, 1))
        #
        self.conv5 = nn.Conv2d(160, 180, kernel_size=( 4 ,4), stride=(2, 2))    
        #
        self.conv6 = nn.Conv2d(180, 200, kernel_size=( 3, 3), stride=(1, 1), padding=( 1, 1))    
        #
        self.conv7 = nn.Conv2d(200, 220, kernel_size=( 3, 3), stride=(1, 1), padding=( 1, 1))    
        #
        self.conv8 = nn.Conv2d(220, 240, kernel_size=( 3, 3), stride=(1, 1), padding=( 1, 1))
        #
        self.conv9 = nn.Conv2d(240, 260, kernel_size=( 2, 2), stride=(2, 2))   
        #
        self.conv10= nn.Conv2d(260, 280, kernel_size=( 2, 2), stride=(1, 1))   
    #


    def forward(self, x):
        #
        # input size = (_, 3, 240, 111)
        x11 = F.leaky_relu(self.inc11(x))
        x12 = F.leaky_relu(self.inc12(x))
        x13 = F.leaky_relu(self.inc13(x))
        x1  = torch.cat((x11, x12, x13), dim=1)  # np.shape(x1) = [_, 30, 240, 11]
        #
        x21 = F.leaky_relu(self.inc21(x1))
        x22 = F.leaky_relu(self.inc22(x1))  
        x23 = F.leaky_relu(self.inc23(x1)) 
        x2  = torch.cat((x21, x22, x23), dim=1)  # np.shape(x2) = [_, 60, 240, 11]
        #
        '''
        # replace first two inception layers with two regular convolutional layers that produce the same number of channels 
        x01= F.leaky_relu(self.conv01(x))
        x02= F.leaky_relu(self.conv02(x01))
        x2 = x02
        '''
        #
        c1 = F.leaky_relu(self.conv1(x2))  # np.shape(c1) = [_, 100, 238, 10]]
        c2 = F.leaky_relu(self.conv2(c1))  # np.shape(c2) = [_, 100, 240, 12]]
        c3 = F.leaky_relu(self.conv3(c2))  # np.shape(c3) = [_, 120, 118, 4]]
        c4 = F.leaky_relu(self.conv4(c3))  # np.shape(c4) = [_, 140, 118, 4]]
        c5 = F.leaky_relu(self.conv5(c4))  # np.shape(c5) = [_, 200, 58, 1]]
        c6 = F.leaky_relu(self.conv6(c5))
        c7 = F.leaky_relu(self.conv7(c6))
        c8 = F.leaky_relu(self.conv8(c7))
        c9 = F.leaky_relu(self.conv9(c8))
        c10= F.leaky_relu(self.conv10(c9))
        #c8 = torch.sigmoid(self.conv8(c7)) # np.shape(c9) = [_, 200, 58, 1]]
        #
        # About the activation function of the last layer:
        # sigmoid:
        #   The last layer passes through a sigmoid activation funtion to ensure that the values in feature map are in the range (0, 1)
        #   It generally takes longer to train the model when using the sigmoid activation funtion.  Set the learning rate to 1e-4.
        #   It might take a few epochs to converge.
        # leaky_relu:
        #   By using leaky_relu as activation function, it is easier to train the model.  Set the learning rate to 1e-3
        #   It converges quicker by using leaky_relu, but the the feature map values could be very large.  (use historgram to check)
        #
        # save the featureMatrix, which is the input for the deconvolution network
        # self.featureMatrix.numel() = _ * 156400
        self.featureMatrix = c10  
        #
        return self.featureMatrix
    #

    def parameterCnt(self):
        pcnt=0
        for p in self.parameters(): 
            pcnt += (p.numel())
        return pcnt


#----------------------------------------------------------------------------------


class Classifier(FeatureExtraction):
    """
    A fully connected network taking feature matrix as input and producing classifications
    """

    def __init__(self):
        super(Classifier, self).__init__()
        #
        # for Linear module weight and bias values initialization, please refer to the pytorch document
        #
        self.fc1 = nn.Linear(280*58*1, 100)
        #
        self.fc2 = nn.Linear(100, 80)
        #
        self.fc3 = nn.Linear(80, 65)



    def forward(self, x):
        fm = super().forward(x)
        fm = fm.view(-1, 280*58*1)
        #
        out1 = F.leaky_relu(self.fc1(fm))
        out2 = F.leaky_relu(self.fc2(out1))
        out3 = F.leaky_relu(self.fc3(out2))
        #
        return out3


    def parameterCnt(self):
        pcnt = super().parameterCnt()
        for p in self.parameters(): 
            pcnt += (p.numel())
        return pcnt
    
    
    