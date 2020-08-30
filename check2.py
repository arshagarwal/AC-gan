import tensorflow as tf
from models.C_gan import Generator,Discriminator
import train_cgan0
import test_cgan
import numpy as np
from PIL import Image
import utils.Image_processing as Image_processing
import torch

a=torch.randint(low=0,high=2,size=(5,))
print(a)