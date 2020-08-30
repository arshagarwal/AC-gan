"""
author: Arsh Agarwal
Code to declare and train the encoders.
"""
import torch


class Encoder_z(torch.nn.Module):
    def __init__(self,l=100):
        """
        :param l: length of latent vector to be obtained
        """
        super(Encoder_z,self).__init__();
        self.l=l
        self.batch_norm1 = torch.nn.BatchNorm2d(32)
        self.batch_norm2 = torch.nn.BatchNorm2d(64)
        self.batch_norm3 = torch.nn.BatchNorm2d(128)
        self.batch_norm4 = torch.nn.BatchNorm2d(256)
        self.batch_norm5 = torch.nn.BatchNorm2d(512)
        self.batch_norm6= torch.nn.BatchNorm1d(4096)

        self.conv_1 = torch.nn.Conv2d(3,32,4,2,padding=1,bias=False)
        self.conv_2 = torch.nn.Conv2d(32,64,4,2,padding=1,bias=False)
        self.conv_3 = torch.nn.Conv2d(64,128,4,2,padding=1,bias=False)
        self.conv_4 = torch.nn.Conv2d(128,256,4,2,padding=1,bias=False)
        self.conv_5=  torch.nn.Conv2d(256,512,4,2,padding=1,bias=False)

        self.relu1 = torch.nn.LeakyReLU(0.2)
        self.relu2 = torch.nn.LeakyReLU(0.2)
        self.relu3 = torch.nn.LeakyReLU(0.2)
        self.relu4 = torch.nn.LeakyReLU(0.2)
        self.relu5 = torch.nn.LeakyReLU(0.2)
        self.relu6 = torch.nn.LeakyReLU(0.2)
        self.relu7 = torch.nn.LeakyReLU(0.2)


        self.flatten=torch.flatten

        self.dense1=torch.nn.Linear(8192,4096,bias=False)
        self.dense2 = torch.nn.Linear(4096,l,bias=False)

    def forward(self,inputs,training=False):
        """
        Gives the encoded output
        :param inputs: Images of the shape [batch_size,128,128,3]
        :param training: Boolean to start or stop training
        :return: the encoded latent vector z
        """
        x = self.conv_1(inputs)
        x = self.batch_norm1(x)
        x=self.relu1(x)

        #second layer
        x = self.conv_2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)

        #3rd layer
        x = self.conv_3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)

        #4th layer
        x = self.conv_4(x)
        x = self.batch_norm4(x)
        x = self.relu4(x)

        #5th layer
        x = self.conv_5(x)
        x = self.batch_norm5(x)
        x = self.relu5(x)


        x = self.flatten(x,start_dim=1)

        assert x.shape == (inputs.shape[0],8192), 'check flattening shape found is {}'.format(x.shape)

        #z vector a.k.a Encoded image vector
        z = self.dense1(x)
        z=self.batch_norm6(z)
        z=self.relu6(z)

        z = self.dense2(z)
        return z
