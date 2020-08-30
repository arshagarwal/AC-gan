from models.t_Cgan import Discriminator,Generator
import torch
import numpy as np
from utils import Image_processing
import tensorflow as tf
import utils.save as save
import os
from models.t_Cgan import Generator
from PIL import Image
import torchvision.transforms as transforms
import torchvision.datasets as dset
import matplotlib.pyplot as plt
import torchvision

def process_3(path,img_size=(128, 128), batch_size=200):
    dataset = dset.ImageFolder(root=path,
                               transform=transforms.Compose([
                                   transforms.Resize(img_size),
                                   transforms.CenterCrop(img_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=2)

    return iter(dataloader)
img_gen = process_3('slim_dataset/Train_dataset')
R_imags,r_y=next(img_gen)
plt.imshow(np.transpose(R_imags[0].numpy(),(1,2,0)))
plt.show()
print(r_y[0])