"""
Author: Arsh Agarwal
Code to test the Cagan trained model
"""
import tensorflow as tf
import numpy as np
from models.C_gan import  Generator,Discriminator
from Options.cgan_test_options import options
import utils.save as save
import utils.Image_processing as Image_processing
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
    optimizer_d = tf.keras.optimizers.Adam(learning_rate=0.0002,beta_1=0.5)
    optimizer_g = tf.keras.optimizers.Adam(learning_rate=0.0002,beta_1=0.5)


    #define and import checkpoints
    checkpoint = tf.train.Checkpoint(Gen=Generator, Discriminator=Discriminator, optimizer1=optimizer_d,
                                     optimizer2=optimizer_g)
    manager = tf.train.CheckpointManager(
        checkpoint, directory="Checkpoints/C_gan_model", max_to_keep=2)
    if(manager.latest_checkpoint):
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        print("restoring from {}th checkpoint".format(manager.latest_checkpoint.split('-')[-1]))
    else:
        raise FileNotFoundError("No checkpoints exist")

    assert mode=='generation' or mode =='augmentation', "{} Mode not available".format(mode)
    if(mode=='generation'):
        # creating the random Y tensor of shape [n_samples,n_classes]
        y = np.zeros(shape=(n_samples,n_classes))
        for i in range(n_samples):
            random_int=np.random.randint(0,n_classes,1)[0]
            y[i,random_int]=1

        # normalizing the y array
        y=y*0.8+0.1
        y=tf.convert_to_tensor(y,dtype=tf.float32)
        labels=Image_processing.get_labels(y)
        z=tf.random.normal(shape=(n_samples,l),mean=0,stddev=1)
        #z=tf.concat([z,y],axis=1)
    else:
        # extract z and corresponding y from z_path, y_path
        pass

    fake_images=Generator(z,labels,False)
    checkpoint_number=manager.latest_checkpoint.split('-')[-1]
    directory='results_'+str(checkpoint_number)
    save.save_images(directory,fake_images)

if __name__=='__main__':
    args=options().parser
    Generator=Generator(ny=args.n_classes,l=args.l)
    Discriminator=Discriminator(args.n_classes)
    test(Generator, Discriminator, n_samples=args.n_samples,n_classes=args.n_classes,
         l=args.l,z_path=args.z_path,y_path=args.y_path,mode=args.mode)








