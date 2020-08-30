"""
Author: Arsh Agarwal

"""
import torch
import numpy as np

class Discriminator(torch.nn.Module):
    def __init__(self,ny):
        super(Discriminator, self).__init__()
        self.ny = ny;

        self.batch_norm2 = torch.nn.BatchNorm2d(128)
        self.batch_norm3 = torch.nn.BatchNorm2d(256)
        self.batch_norm4 = torch.nn.BatchNorm2d(512)
        self.batch_norm5 = torch.nn.BatchNorm2d(1024)

        self.relu1 = torch.nn.LeakyReLU(0.2)
        self.relu2 = torch.nn.LeakyReLU(0.2)
        self.relu3 = torch.nn.LeakyReLU(0.2)
        self.relu4 = torch.nn.LeakyReLU(0.2)
        self.relu5 = torch.nn.LeakyReLU(0.2)

        self.conv1 = torch.nn.Conv2d(3,64,4,(2,2),padding=1,bias=False)
        self.conv2 = torch.nn.Conv2d(64,128,4,2,padding=1,bias=False)
        self.conv3 = torch.nn.Conv2d(128,256,4,2,padding=1,bias=False)
        self.conv4 = torch.nn.Conv2d(256,512,4,2,padding=1,bias=False)
        self.conv5 = torch.nn.Conv2d(512,1024,4,2,padding=1,bias=False)

        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0.5)
        self.dropout3 = torch.nn.Dropout(0.5)
        self.dropout4 = torch.nn.Dropout(0.5)
        self.dropout5 = torch.nn.Dropout(0.5)


        self.o1=torch.nn.Conv2d(1024,1,4,1,0,bias=False)
        self.o2 = torch.nn.Conv2d(1024, ny, 4, 1, 0, bias=False)

        self.sigmoid=torch.nn.Sigmoid()
        self.soft=torch.nn.Softmax(dim=1)

    def forward(self, z):
        """
        Performs both real fake discrimination and categorical classification
        :param z:  Images of the shape [batch_size,128,128,3]
        :return:
        """
        x = self.conv1(z)
        x = self.relu1(x)
        x = self.dropout1(x)

        assert x.shape==(x.shape[0],64,64,64),"{}".format(x.shape)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu2(x)
        x = self.dropout2(x)


        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu3(x)
        x = self.dropout3(x)


        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.relu4(x)
        x = self.dropout4(x)


        x = self.conv5(x)
        x = self.batch_norm5(x)
        x = self.relu5(x)
        x = self.dropout5(x)


        assert x.shape==(x.shape[0],1024,4,4)

        o1 = self.o1(x)
        o1=torch.reshape(o1,(x.shape[0],1))
        o1=self.sigmoid(o1)
        o2 = self.o2(x)
        o2=torch.reshape(o2,(x.shape[0],self.ny))
        o2=self.soft(o2)

        return o1, o2

class Generator(torch.nn.Module):
    """Given latent vector, produces Image of the shape (128,128,3)
      latent vector shape: [batch_size,z+ny]
      where
      z: is the number of parameters into which which image is encoded using Encoder_z (100)
      ny:number of attributes in an image
      """
    def __init__(self,ny,l):
        """

        :param l: number of units in the latent vector
        :param ny: number of categories
        """
        super(Generator,self).__init__();
        self.l=l

        self.conv = torch.nn.ConvTranspose2d(self.l,1024,4,1,0,bias=False)


        # embedding layer
        self.embedding =torch.nn.Embedding(ny,50)
        self.dense2 = torch.nn.Linear(50,4*4,bias=False)



        # conv layers
        self.conv0 = torch.nn.ConvTranspose2d(1025,512,4,2,1,bias=False)
        self.conv1 = torch.nn.ConvTranspose2d(512,256,4,2,1,bias=False)

        self.conv2 = torch.nn.ConvTranspose2d(256,128,4,2,1,bias=False)
        self.conv3 = torch.nn.ConvTranspose2d(128,64,4,2,1,bias=False)
        self.conv4 = torch.nn.ConvTranspose2d(64,3,4,2,1,bias=False)

        #tanh activation
        self.tanh=torch.nn.Tanh()

        # batch_norm layers
        self.batch_norm0=torch.nn.BatchNorm2d(512)
        self.batch_norm1=torch.nn.BatchNorm2d(256)
        self.batch_norm2 = torch.nn.BatchNorm2d(128)
        self.batch_norm3 = torch.nn.BatchNorm2d(64)
        self.conv_batch_norm=torch.nn.BatchNorm2d(1024)

        # leaky relu layers
        self.relu1 = torch.nn.LeakyReLU(0.2)
        self.relu2 = torch.nn.LeakyReLU(0.2)
        self.relu3 = torch.nn.LeakyReLU(0.2)
        self.relu4 = torch.nn.LeakyReLU(0.2)
        self.relu5 = torch.nn.LeakyReLU(0.2)









    def forward(self,input,labels):
        """
        Generates synthetic images of the shape [batch_size,128,128,3]
        :param input: Latent vector + attributes of the shape [batch_size,z+l]
        :return: synthetic images of the shape [batch_size,128,128,3]
        """
        # creating the embedding
        embedding=self.embedding(labels)
        assert embedding.shape == (input.shape[0], 50), " check generator embedding shape {} ".format(embedding.shape)
        embedding=self.dense2(embedding)
        embedding=torch.reshape(embedding,(input.shape[0],1,4,4))

        input = torch.reshape(input, (input.shape[0],self.l,1,1) )
        x=self.conv(input)
        x = self.conv_batch_norm(x)
        x=self.relu1(x)


        assert x.shape==(input.shape[0],1024,4,4), 'check generators first conv layer'

        # concatenating embedding
        x=torch.cat((x,embedding),axis=1);

        assert x.shape == (input.shape[0],1025,4,4), "check embedding concatenation"

        x = self.conv0(x)
        x = self.batch_norm0(x)
        x = self.relu2(x)

        x=self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu3(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu4(x)


        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.relu5(x)


        x = self.conv4(x)

        return x
















