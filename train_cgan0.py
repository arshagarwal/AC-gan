"""
Author: Arsh Agarwal
 To Train C-gan network
"""
import tensorflow as tf
from Options.cgan_train_options import options
import utils.Image_processing as Image_processing

from models.C_gan import Generator,Discriminator
import time
import numpy as np
import os

def train(Generator,Discriminator,img_gen,n_images,epochs=100,batch_size=20,lambda_A=2,lambda_B=1,loss_c='cce',mode='generation',l=100,continue_train=False,save_freq=10, max_save=5):
    """

    :param Generator: Instance of Generator that is to be trained
    :param Discriminator: Instance of Discriminator to be trained
    :param img_gen: Image generator object produced by utils.process function
    :param n_images: total number of images
    :param epochs: number of epochs
    :param batch_size: number of images that undergo forward/backward prop at a time.
    :param lambda_A: factor by which discriminator loss is divided
    :param lambda_B: factor by which attribute classification loss is multiplied w.r.t real/fake loss.
    :param loss_c: loss function for attribute classification, can take the following values ('cce' : categoricalcrossentropy | 'mse' : minimum squared error, 'mae' : minimum absolute error )
    :param mode: specifies the mode of training as either 'generation' or 'augmentation'
    :param l: number of units in noise
    :param save_freq: freq with which checkpoints are saved
    :param continue_train: Boolaen that denotes whether to start from previous checkpoint or start from scratch
    :param max_save: max checkpoints to be saved
    :return: void
    """
    n_batches = (int)(n_images / batch_size);

    # defining the gen and disc optimizers
    optimizer_d = tf.keras.optimizers.Adam(learning_rate=0.0002,beta_1=0.5)
    optimizer_g = tf.keras.optimizers.Adam(learning_rate=0.0002,beta_1=0.5)



    start_epoch=1;

    # defining checkpoints
    checkpoint = tf.train.Checkpoint(Gen=Generator, Discriminator=Discriminator,optimizer1=optimizer_d,optimizer2=optimizer_g)
    manager = tf.train.CheckpointManager(
        checkpoint, directory="Checkpoints/C_gan_model", max_to_keep=max_save)
    if continue_train:
        status = checkpoint.restore(manager.latest_checkpoint)
        if(manager.latest_checkpoint):

            start_epoch=(int)(manager.latest_checkpoint.split('-')[-1])+1
            print("starting from {}th checkpoints".format(start_epoch))
        else:
            raise ImportError("No checkpoint found ")

    for i in range(start_epoch,start_epoch+epochs):

        # generator and discriminator category classification losses
        gen_t_c=0
        disc_t_c=0
        # generator and discriminator real/ fake discrimination loss
        gen_d=0
        disc_d=0
        start_time=time.time()




        for j in range(n_batches):

            # real images and real Y(only attributes)
            (R_images, R_Y) = next(img_gen)
            R_images = tf.convert_to_tensor(R_images)
            R_Y = tf.convert_to_tensor(R_Y)

            if R_images.shape[0]!=batch_size:
                (R_images, R_Y) = next(img_gen)
                R_images = tf.convert_to_tensor(R_images)
                R_Y = tf.convert_to_tensor(R_Y)

            assert R_images.shape[0] == batch_size, "Number of images sampled were not equal to batch size"

            # generator training step
            with tf.GradientTape() as tape:
                tape.watch(Generator.trainable_variables)
                # sampling encodings from z vector
                if mode=='generation':
                    curr_z = tf.random.normal(shape=(batch_size,l),mean=0.0,stddev=1.0)
                    curr_y=tf.random.shuffle(R_Y)
                    labels=Image_processing.get_labels(curr_y)
                    #curr_z=tf.concat([curr_z,curr_y],axis=1)
                else:
                    pass
                fake_images = Generator(curr_z,labels,True)

                #  y_pred1 is the real/fake loss and y_pred2 is the attribute classification loss
                y_pred1, y_pred2 = Discriminator(fake_images,True)

                y_true=tf.ones(shape=(fake_images.shape[0],1))
                # Slicing the feature vectors from z
                y_z_true=curr_y

                #real/fake loss
                assert y_true.shape==y_pred1.shape, "y_true shape: {} y_pred1 shape: {} ".format(y_true.shape,y_pred1.shape)
                loss1=tf.keras.losses.binary_crossentropy(y_true,y_pred1)
                gen_d+=tf.reduce_sum(loss1)

                # attribute classification loss
                assert y_z_true.shape == y_pred2.shape, "y_z_true shape: {} y_pred2 shape: {} ".format(y_z_true.shape, y_pred2.shape)
                if loss_c == 'cce':
                    loss2 = tf.keras.losses.categorical_crossentropy(y_z_true,y_pred2)
                elif loss_c == 'mse':
                    loss2=tf.keras.losses.mean_squared_error(y_z_true,y_pred2)
                elif loss_c == 'mae':
                    loss2 = tf.keras.losses.mean_absolute_error(y_z_true, y_pred2)
                else:
                    raise ValueError("Given loss not recognized")
                gen_t_c+=tf.reduce_sum(lambda_B*loss2)
                loss=loss1+lambda_B*loss2;

            grads = tape.gradient(loss,Generator.trainable_variables)
            optimizer_g.apply_gradients(zip(grads,Generator.trainable_variables))


            #discriminator training step
            with tf.GradientTape() as tape:
                tape.watch(Discriminator.trainable_variables)


                real_img = R_images
                fake_img = fake_images;

                # calculating loss for real samples
                y_true = tf.ones(shape=(real_img.shape[0],1))
                y_z_true=R_Y
                y_pred1,y_pred2 = Discriminator(real_img,True)
                assert y_true.shape == y_pred1.shape, "y_true shape: {} y_pred1 shape: {} ".format(y_true.shape,y_pred1.shape)
                loss1 = tf.keras.losses.binary_crossentropy(y_true, y_pred1)

                # attribute classification loss
                assert y_z_true.shape == y_pred2.shape, "y_z_true shape: {} y_pred2 shape: {} ".format(y_z_true.shape,
                                                                                                   y_pred2.shape)
                if (loss_c == 'cce'):
                    loss2 = tf.keras.losses.categorical_crossentropy(y_z_true, y_pred2)
                elif (loss_c == 'mse'):
                    loss2 = tf.keras.losses.mean_squared_error(y_z_true, y_pred2)
                elif (loss_c == 'mae'):
                    loss2 = tf.keras.losses.mean_absolute_error(y_z_true, y_pred2)
                else:
                    raise ValueError("Given loss not recognized")

                #total real loss
                loss_t_r=loss1+lambda_B*loss2;
                disc_d+=tf.reduce_sum(loss1)
                disc_t_c=tf.reduce_sum(lambda_B*loss2)

                # calculating loss for fake images
                y_z_fake_true=curr_y
                y_pred3,y_pred4=Discriminator(fake_img,True)


                y_fake_true=tf.zeros(shape=(fake_img.shape[0],1))
                assert y_fake_true.shape == y_pred3.shape, "y_fake_true shape: {} y_pred3 shape: {} ".format(y_fake_true.shape,y_pred3.shape)
                loss3=tf.keras.losses.binary_crossentropy(y_fake_true, y_pred3)

                # attribute classification loss
                assert y_z_fake_true.shape == y_pred4.shape, "y_z_fake_true shape: {} y_pred4 shape: {} ".format(y_z_fake_true.shape,y_pred4.shape)
                if (loss_c == 'cce'):
                    loss4 = tf.keras.losses.categorical_crossentropy(y_z_fake_true, y_pred4)
                elif (loss_c == 'mse'):
                    loss4 = tf.keras.losses.mean_squared_error(y_z_fake_true, y_pred4)
                elif (loss_c == 'mae'):
                    loss4 = tf.keras.losses.mean_absolute_error(y_z_fake_true, y_pred4)
                else:
                    raise ValueError("Given loss not recognized")

                assert tf.reduce_sum(lambda_B * loss4)!=0, "classification loss is zero"
                #total fake loss
                loss_t_f = loss3 + lambda_B*loss4;
                #total loss
                loss_t = tf.concat([loss_t_r,loss_t_f],axis=0)
                loss_t=loss_t/lambda_A;
                assert (loss_t.shape[0] == 2*batch_size ), 'Check Discriminator training step'

                disc_d += tf.reduce_sum(loss3)
                disc_t_c = tf.reduce_sum(lambda_B * loss4)

            grads = tape.gradient(loss_t,Discriminator.trainable_variables)
            optimizer_d.apply_gradients(zip(grads,Discriminator.trainable_variables))

        time_taken=round(time.time()-start_time)
        # dividing losses by n_batches so as to produce loss per image
        x=n_batches * batch_size
        disc_d=disc_d/x
        gen_d=gen_d/x
        disc_t_c=disc_t_c/x
        gen_t_c=gen_t_c/x
        print("Epoch no: {} time : {}s  Discriminator_D loss: {} Generator_D loss: {} Discriminator_C loss: {} Generator_C loss: {}"
              .format(i,time_taken,disc_d,gen_d,disc_t_c,gen_t_c))

        if i % save_freq == 0 or i==(start_epoch+epochs-1):
            print("saving checkpoint for {}th epoch".format(i))
            manager.save(checkpoint_number=i)


if __name__ == '__main__':
    args= options().parser
    images,y = Image_processing.process(args.data_path)
    Generator = Generator(l=args.l,ny=y.shape[1])
    Discriminator = Discriminator(ny=y.shape[1])
    z=[]
    img_gen=Image_processing.process2(args.data_path,batch_size=args.batch_size)
    n_images = 0
    dirnames = os.listdir(args.data_path)
    for i in dirnames:
        n_images+= len(os.listdir(args.data_path+'/'+i))
    print("Total number of images: {}".format(n_images))
    if(args.mode=="augmentation"):
        file_path=args.z_file
        # create z vector here
        z=Image_processing.create_z(file_path)
        pass

    train(Generator,Discriminator,img_gen=img_gen,n_images=n_images,epochs=args.epochs,batch_size=args.batch_size,lambda_A=args.lambda_A,
        lambda_B=args.lambda_B,loss_c=args.loss_c,mode=args.mode,l=args.l,save_freq=args.save_freq,continue_train=args.continue_train)

