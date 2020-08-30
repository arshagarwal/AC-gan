"""
Author: Arsh Agarwal
code to save images and tensors
"""
import os
from PIL import Image
import torch
import numpy as np

def save_images(directory,images):
    """
    saves images in the given directory
    :param directory: name of directory in which images are to be saved
    :return: void
    """
    # de-normalizing
    images= images*127.5+127.5;
    folders=os.listdir()
    if('results' not in folders):
        os.mkdir(directory)
    for im in range(len(images)):
        curr=images[im].detach().cpu().numpy()
        curr=np.transpose(curr,[1,2,0])
        pil_img = Image.fromarray(curr.astype('uint8'))
        address = directory+"/"+str(im)+'.png'
        pil_img.save(address)
    print("Images saved in {} directory".format(directory))
    return

def save_checkpoints(gen,disc,opt_g,opt_d,epoch):
    """
    checks for checkpoint directory and saves checkpoint in the directory
    needed only for torch implementation
    :param gen: generator instance
    :param disc: discriminator instance
    :param opt_g: optimizer_g instance
    :param opt_d: optimizer_d instance
    """
    if('Checkpoints' not in os.listdir()):
        os.mkdir('Checkpoints')
    if(len(os.listdir('Checkpoints'))>=1):
        os.remove('Checkpoints/'+os.listdir('Checkpoints')[-1])
    path='Checkpoints/checkpoint_'+str(epoch)+'.pt'
    torch.save({
        'generator':gen.state_dict(),
        'discriminator':disc.state_dict(),
        'optimizer_g': opt_g.state_dict(),
        'optimizer_d':opt_d.state_dict(),
        'epoch':epoch
    },path)
def save_encoder_checkpoints(encoder,epoch,optimizer):
    """
    checks for checkpoint directory and saves checkpoint in the directory
    needed only for torch implementation
    :param gen: generator instance
    :param disc: discriminator instance
    :param opt_g: optimizer_g instance
    :param opt_d: optimizer_d instance
    """
    if('Encoder_Checkpoints' not in os.listdir()):
        os.mkdir('Encoder_Checkpoints')
    if(len(os.listdir('Encoder_Checkpoints'))>=1):
        os.remove('Encoder_Checkpoints/'+os.listdir('Encoder_Checkpoints')[-1])
    path='Encoder_Checkpoints/checkpoint_'+str(epoch)+'.pt'
    torch.save({
        'encoder':encoder.state_dict(),
        'optimizer':optimizer.state_dict(),
        'epoch':epoch
    },path)

