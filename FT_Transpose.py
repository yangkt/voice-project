import torch
import torch.nn as nn
import torch.nn.functional as F



class FeatureTranspose(nn.Module):
    """
    A transposed CNN network (deconvolution network) to reconstruct the input
    """
    
    def __init__(self):
        super(FeatureTranspose, self).__init__()
        #
        self.tinc01 = nn.ConvTranspose2d(280,  70, kernel_size=( 1, 1), stride=1, padding=(0, 0))
        self.tinc02 = nn.ConvTranspose2d(280,  70, kernel_size=( 5, 1), stride=1, padding=(2, 0))
        self.tinc03 = nn.ConvTranspose2d(280,  70, kernel_size=( 9, 1), stride=1, padding=(4, 0))
        self.tinc04 = nn.ConvTranspose2d(280,  70, kernel_size=(13, 1), stride=1, padding=(6, 0))
        #
        self.tinc11 = nn.ConvTranspose2d(280,  80, kernel_size=( 5, 1), stride=1, padding=(0, 0)) # 80,  60-2*0,  1
        self.tinc12 = nn.ConvTranspose2d(280,  80, kernel_size=( 9, 1), stride=1, padding=(2, 0)) # 80,  64-2*2,  1
        self.tinc13 = nn.ConvTranspose2d(280,  80, kernel_size=(13, 1), stride=1, padding=(4, 0)) # 80,  68-2*4,  1
        #
        self.tinc21 = nn.ConvTranspose2d(240,  50, kernel_size=( 5, 1), stride=1, padding=(0, 0)) # 50,  66-2*0,  1
        self.tinc22 = nn.ConvTranspose2d(240,  50, kernel_size=( 7, 1), stride=1, padding=(1, 0)) # 50,  68-2*1,  1
        self.tinc23 = nn.ConvTranspose2d(240,  50, kernel_size=( 9, 1), stride=1, padding=(2, 0)) # 50,  70-2*2,  1        
        self.tinc24 = nn.ConvTranspose2d(240,  50, kernel_size=(11, 1), stride=1, padding=(3, 0)) # 50,  72-2*3,  1        
        #
        self.tconv1 = nn.ConvTranspose2d(200, 180, kernel_size=( 9, 1), stride=1)  # 180,  74,  1 
        #
        self.tconv2 = nn.ConvTranspose2d(180, 150, kernel_size=( 9, 1), stride=1)  # 150,  82,  1 
        #
        self.tconv3 = nn.ConvTranspose2d(150, 120, kernel_size=( 9, 1), stride=1)  # 120,  90,  1
        #
        self.tconv4 = nn.ConvTranspose2d(120, 100, kernel_size=( 9, 1), stride=1)  # 100,  98,  1
        #
        self.tconv5 = nn.ConvTranspose2d(100,  80, kernel_size=( 4, 1), stride=2)  #  80, 198,  1
        #
        self.tconv6 = nn.ConvTranspose2d( 80,  40, kernel_size=( 9, 1), stride=1)  #  40, 206,  1 
        #
        self.tconv7 = nn.ConvTranspose2d( 40,  20, kernel_size=( 9, 1), stride=1)  #  20, 214,  1 
        #
        self.tconv8 = nn.ConvTranspose2d( 20,  10, kernel_size=( 9, 1), stride=1)  #  10, 222,  1 
        #
        self.tconv9 = nn.ConvTranspose2d( 10,   5, kernel_size=( 9, 1), stride=1)  #   5, 230,  1 
        #
        self.tconv10= nn.ConvTranspose2d(  5,   1, kernel_size=(11, 1), stride=1)  #   1, 240,  1 
        #
        self.sigmoid = nn.Sigmoid()
    #


    def forward(self, x):
        #
        # input size = (_, 280, 58, 1)
        x01 = F.leaky_relu(self.tinc01(x))
        x02 = F.leaky_relu(self.tinc02(x))  
        x03 = F.leaky_relu(self.tinc03(x)) 
        x04 = F.leaky_relu(self.tinc04(x)) 
        x0  = torch.cat((x01, x02, x03, x04), dim=1)  # np.shape(x0) = [_, 280, 58, 1]
        #
        x11 = F.leaky_relu(self.tinc11(x0))
        x12 = F.leaky_relu(self.tinc12(x0))
        x13 = F.leaky_relu(self.tinc13(x0))
        x1  = torch.cat((x11, x12, x13), dim=1)  # np.shape(x1) = [_, 240, 62, 1]
        #
        x21 = F.leaky_relu(self.tinc21(x1))
        x22 = F.leaky_relu(self.tinc22(x1))  
        x23 = F.leaky_relu(self.tinc23(x1)) 
        x24 = F.leaky_relu(self.tinc24(x1)) 
        x2  = torch.cat((x21, x22, x23, x24), dim=1)  # np.shape(x2) = [_, 200, 66, 1]
        #
        #
        tc1 = F.leaky_relu(self.tconv1(x2))    
        tc2 = F.leaky_relu(self.tconv2(tc1))  
        tc3 = F.leaky_relu(self.tconv3(tc2))  
        tc4 = F.leaky_relu(self.tconv4(tc3))  
        tc5 = F.leaky_relu(self.tconv5(tc4))  
        tc6 = F.leaky_relu(self.tconv6(tc5))  
        tc7 = F.leaky_relu(self.tconv7(tc6))  
        tc8 = F.leaky_relu(self.tconv8(tc7))  
        tc9 = F.leaky_relu(self.tconv9(tc8)) 
        #
        #tc10_=self.tconv10(tc9)
        #tc10= torch.mul(torch.tanh(tc10_), F.relu(tc10_))
        tc10= F.relu(self.tconv10(tc9)) 
        #
        self.transposed = tc10
        #
        return self.transposed
    #


    def parameterCnt(self):
        pcnt=0
        for p in self.parameters(): 
            pcnt += (p.numel())
        return pcnt
    #
    

#----------------------------------------------------------------------------------


class AudioGenerator(FeatureTranspose):
    def __init__(self):
        super(AudioGenerator, self).__init__()
    #


    def forward(self, features):
        return super().forward(features)
    #


    def parameterCnt(self):
        pcnt = super().parameterCnt()
        for p in self.parameters(): 
            pcnt += (p.numel())
        return pcnt
    
"""
x = torch.randn(1, 280, 58, 1)
ag = AudioGenerator()
amp = ag.forward(x)

ag.parameterCnt()
"""
    







    
    
    