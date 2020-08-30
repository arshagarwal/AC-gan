"""
Author: Arsh Agarwal
Code to test the Cagan trained model
"""
import torch
import numpy as np
from models.t_Cgan import  Generator,Discriminator
from Options.cgan_test_options import options
import utils.save as save
import utils.Image_processing as Image_processing
import os
def test(Generator,Discriminator,n_classes,n_samples=50,l=100,z_path="",y_path="",mode='generation'):
    """
    tests the model with the latest checkpoint and
    crates a directory where synthetic images produced  are stored

    :param Generator: Instance of the Genearator model
    :param Discriminator: Instance of the discriminator model
    :param n_samples: no of images to be produced
    :param n_classes: number of classes
    :param l: size of the latent vector
    :param z_path: path to z tensors, needed for augmentation mode
    :param y_path: path to corresponding y tensors, needed in augmentation mode
    :param mode: String to specify the mode
    :return: void
    """

    # defining the gen and disc optimizers
    optimizer_d = torch.optim.Adam(Discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_g = torch.optim.Adam(Generator.parameters(), lr=0.0002, betas=(0.5, 0.999))


    #define and import checkpoints
    assert 'Checkpoints' in os.listdir(), "No saved checkpoints found"
    path = 'Checkpoints/' + os.listdir('Checkpoints')[0]
    checkpoint = torch.load(path)

    Generator.load_state_dict(checkpoint['generator'])
    Discriminator.load_state_dict(checkpoint['discriminator'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g'])

    start_epoch = checkpoint['epoch'] + 1;
    print("starting from {}th epoch".format(start_epoch))
    assert mode=='generation' or mode =='augmentation', "{} Mode not available".format(mode)
    if(mode=='generation'):
        # creating the random Y tensor of shape [n_samples,n_classes]
        y = np.zeros(shape=(n_samples,n_classes))
        for i in range(n_samples):
            random_int=np.random.randint(0,n_classes,1)[0]
            y[i,random_int]=1

        # normalizing the y array
        y=y*0.8+0.1
        labels=Image_processing.get_labels(y)
        labels=torch.from_numpy(labels).long()
        z=torch.randn((n_samples,l))
        #z=tf.concat([z,y],axis=1)
    else:
        # extract z and corresponding y from z_path, y_path
        pass
    Generator.eval()
    fake_images=Generator(z,labels)
    checkpoint_number=checkpoint['epoch']
    directory='results_'+str(checkpoint_number)
    save.save_images(directory,fake_images)

if __name__=='__main__':
    args=options().parser
    Generator=Generator(ny=args.n_classes,l=args.l)
    Discriminator=Discriminator(args.n_classes)
    test(Generator, Discriminator, n_samples=args.n_samples,n_classes=args.n_classes,
         l=args.l,z_path=args.z_path,y_path=args.y_path,mode=args.mode)








