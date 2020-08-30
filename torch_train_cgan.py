"""
Author: Arsh Agarwal
 To Train C-gan network
"""
import torch
from Options.cgan_train_options import options
import utils.Image_processing as Image_processing
from models.t_Cgan import Generator,Discriminator
import time
import utils.save as save
import numpy as np
import os

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0)
    if type(m)==torch.nn.Embedding:
        torch.nn.init.normal_(m.weight.data,mean=0.0,std=0.02)
    elif type(m)==torch.nn.Linear:
        torch.nn.init.normal_(m.weight.data,0,0.02)

def train(Generator,Discriminator,n_images,epochs=100,batch_size=20,lambda_A=2,lambda_B=1,loss_c='cce',mode='generation',l=100,continue_train=False,save_freq=10, max_save=5,datapath='slim_dataset/Train_dataset'):
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
    :param datapath: path to the dataset file
    :return: void
    """
    n_batches = (int)(n_images / batch_size);

    # defining the gen and disc optimizers
    optimizer_d = torch.optim.Adam(Discriminator.parameters(),lr=0.0002,betas=(0.5, 0.999))
    optimizer_g = torch.optim.Adam(Generator.parameters(),lr=0.0002,betas=(0.5, 0.999))

    if torch.cuda.is_available():
        device=torch.device('cuda:0')
    else:
        device=torch.device('cpu')



    start_epoch=1;
    if continue_train:
        assert 'Checkpoints' in os.listdir() ,"No saved checkpoints found"
        path='Checkpoints/'+os.listdir('Checkpoints')[0]
        checkpoint=torch.load(path,map_location=device)

        Generator.load_state_dict(checkpoint['generator'])
        Discriminator.load_state_dict(checkpoint['discriminator'])
        Generator.to(device)
        Discriminator.to(device)
        optimizer_d = torch.optim.Adam(Discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_g = torch.optim.Adam(Generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_d.load_state_dict(checkpoint['optimizer_d'])
        optimizer_g.load_state_dict(checkpoint['optimizer_g'])

        start_epoch=checkpoint['epoch']+1;
        print("starting from {}th epoch".format(start_epoch))
    else:
        # initializing weights
        Generator.apply(weights_init)
        Discriminator.apply(weights_init)
        Generator.to(device)
        Discriminator.to(device)

    print("device is {}".format(device))
    Generator.train()
    Discriminator.train()
    for i in range(start_epoch,start_epoch+epochs):

        # generator and discriminator category classification losses
        gen_t_c=0
        disc_t_c=0
        # generator and discriminator real/ fake discrimination loss
        gen_d=0
        disc_d=0
        start_time=time.time()

        img_gen=Image_processing.process_3(datapath,batch_size=batch_size)


        for j in range(n_batches):

            Generator.zero_grad()

            # real images and real Y(only attributes)
            R_images, R_Y = next(img_gen)

            assert R_images.shape[0]==batch_size, "check image generator"

            # sampling encodings from z vector
            if mode=='generation':
                curr_z=torch.randn((batch_size,l),device=device)
                curr_y=np.copy(R_Y);
                np.random.shuffle(curr_y)
                labels =R_Y.detach().clone().numpy()
                np.random.shuffle(labels)

                curr_y=torch.from_numpy(curr_y).to(device)
                labels=torch.from_numpy(labels).long().to(device)
                #curr_z=tf.concat([curr_z,curr_y],axis=1)
            else:
                pass
            fake_images = Generator(curr_z,labels)

            # discriminator training step
            real_img = R_images.to(device)
            fake_img = fake_images;
            Discriminator.zero_grad()
            # calculating loss for real samples
            y_true_r = torch.ones((real_img.shape[0],1)).to(device)
            y_z_true_r=R_Y.to(device)
            y_pred1_r,y_pred2_r = Discriminator(real_img)
            assert y_true_r.shape == y_pred1_r.shape, "y_true shape: {} y_pred1 shape: {} ".format(y_true_r.shape,y_pred1_r.shape)
            loss1_r = torch.nn.BCELoss()(y_pred1_r,y_true_r)

            # attribute classification loss
            assert y_z_true_r.shape[0] == y_pred2_r.shape[0], "y_z_true shape: {} y_pred2 shape: {} ".format(y_z_true_r.shape,
                                                                                               y_pred2_r.shape)
            if (loss_c == 'cce'):
                loss2_r = torch.nn.CrossEntropyLoss()(y_pred2_r, y_z_true_r)
            elif (loss_c == 'mse'):
                loss2_r=torch.nn.MSELoss()(y_pred2_r,y_z_true_r)
            elif (loss_c == 'mae'):
                loss2_r = torch.nn.L1Loss()(y_pred2_r, y_z_true_r)
            else:
                raise ValueError("Given loss not recognized")

            #total real loss
            loss_t_r=loss1_r+lambda_B*loss2_r;
            loss_t_r.backward()
            disc_d+=loss1_r
            disc_t_c+=lambda_B*loss2_r

            # calculating loss for fake images
            y_z_fake_true=labels
            y_pred3,y_pred4=Discriminator(fake_img.detach())


            y_fake_true=torch.zeros((fake_img.shape[0],1),device=device)
            assert y_fake_true.shape == y_pred3.shape, "y_fake_true shape: {} y_pred3 shape: {} ".format(y_fake_true.shape,y_pred3.shape)
            loss3=torch.nn.BCELoss()(y_pred3,y_fake_true)


            # attribute classification loss
            assert y_z_fake_true.shape[0] == y_pred4.shape[0], "y_z_fake_true shape: {} y_pred4 shape: {} ".format(y_z_fake_true.shape,y_pred4.shape)
            if (loss_c == 'cce'):
                loss4 = torch.nn.CrossEntropyLoss()(y_pred4, y_z_fake_true)

            elif (loss_c == 'mse'):
                loss4 = torch.nn.MSELoss()(y_pred4, y_z_fake_true)

            elif (loss_c == 'mae'):
                loss4 = torch.nn.L1Loss()(y_pred4, y_z_fake_true)
            else:
                raise ValueError("Given loss not recognized")

            assert lambda_B * loss4!=0, "classification loss is zero"
            #total fake loss
            loss_t_f = (loss3 + lambda_B*loss4);
            loss_t_f.backward()
            optimizer_d.step()

            disc_d += loss3
            disc_t_c += lambda_B * loss4


            #generator training step
            # real/fake loss
            #  y_pred1 is the real/fake loss and y_pred2 is the attribute classification loss
            y_pred1, y_pred2 = Discriminator(fake_images)

            y_true = torch.ones((fake_images.shape[0], 1), device=device)
            # Slicing the feature vectors from z
            y_z_true = curr_y
            assert y_true.shape == y_pred1.shape, "y_true shape: {} y_pred1 shape: {} ".format(y_true.shape,
                                                                                               y_pred1.shape)
            loss1 = torch.nn.BCELoss()(y_pred1, y_true)
            gen_d += loss1

            # attribute classification loss
            assert labels.shape[0] == y_pred2.shape[0], "y_z_true shape: {} y_pred2 shape: {} ".format(y_z_true.shape,
                                                                                                   y_pred2.shape)
            if loss_c == 'cce':
                loss2 = torch.nn.CrossEntropyLoss()(y_pred2, labels)
            elif loss_c == 'mse':
                loss2 = torch.nn.MSELoss()(y_pred2, y_z_true)
            elif loss_c == 'mae':
                loss2 = torch.nn.L1Loss()(y_pred2, y_z_true)
            else:
                raise ValueError("Given loss not recognized")
            gen_t_c += lambda_B * loss2
            loss = loss1 + lambda_B * loss2;


            loss.backward()
            optimizer_g.step()



        time_taken=round(time.time()-start_time)
        # dividing losses by n_batches so as to produce loss per image
        x=n_batches
        disc_d=disc_d/x
        gen_d=gen_d/x
        disc_t_c=disc_t_c/x
        gen_t_c=gen_t_c/x
        disc_d = disc_d.item()
        gen_d = gen_d.item()
        disc_t_c = disc_t_c.item()
        gen_t_c = gen_t_c.item()
        print("Epoch no: {} time : {}s  Discriminator_D loss: {} Generator_D loss: {} Discriminator_C loss: {} Generator_C loss: {}"
              .format(i,time_taken,disc_d,gen_d,disc_t_c,gen_t_c))

        if i % save_freq == 0 or i==(start_epoch+epochs-1):
            save.save_checkpoints(Generator,Discriminator,optimizer_g,optimizer_d,i)
            print("saving checkpoint for {}th epoch".format(i))



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

    train(Generator,Discriminator,n_images=n_images,epochs=args.epochs,batch_size=args.batch_size,lambda_A=args.lambda_A,
        lambda_B=args.lambda_B,loss_c=args.loss_c,mode=args.mode,l=args.l,save_freq=args.save_freq,continue_train=args.continue_train,datapath=args.data_path)

