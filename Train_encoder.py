"""
Author: Arsh Agarwal
Code to train the encoder
"""
from models.t_Cgan import Generator
from models.t_encoder import Encoder_z
import os
import utils.save as save
import torch
import time
from Options.encoder_train_options import options

def loss_function(pred,target):
    loss=torch.abs(pred-target)
    loss=torch.square(loss)
    return torch.sum(loss);

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


def train(Generator,Encoder_z,continue_train=False,epochs=100,batch_size=50,n_batches=100,l=100,save_freq=10,ny=2):


    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')

    # loading the generator checkpoints
    assert 'Checkpoints' in os.listdir(), "No saved checkpoints found"
    path = 'Checkpoints/' + os.listdir('Checkpoints')[0]
    checkpoint = torch.load(path)
    Generator.load_state_dict(checkpoint['generator'])
    Generator=Generator.to(device)
    Generator.eval()

    optimizer = torch.optim.Adam(Encoder_z.parameters(),lr=0.0002,betas=(0.5, 0.999))

    start_epochs=1
    if continue_train:
        path='Encoder_Checkpoints/'+os.listdir('Encoder_Checkpoints')[0]
        checkpoint=torch.load(path)
        Encoder_z.load_state_dict(checkpoint['encoder'])
        Encoder_z=Encoder_z.to(device)
        optimizer=optimizer = torch.optim.Adam(Encoder_z.parameters(),lr=0.0002,betas=(0.5, 0.999))
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epochs=checkpoint['epoch']+1
        print("starting from {}th checkpoint".format(start_epochs))
    else:
       Encoder_z.apply(weights_init)
       Encoder_z = Encoder_z.to(device)


    fixed_noise=torch.randn((n_batches,batch_size,l),device='cpu')
    # random integers from the set [0,2)
    labels=torch.randint(low=0,high=2,size=(n_batches,batch_size))

    for i in range(start_epochs,start_epochs+epochs):
        loss=0
        start_time=time.time()
        for j in range(n_batches):

            # sampling noise and labels
            curr_noise=fixed_noise[i]
            curr_labels=labels[i]
            curr_noise=curr_noise.to(device)
            curr_labels=curr_labels.to(device)

            # generating fake images
            fake_images=Generator.forward(curr_noise,curr_labels)

            Encoder_z.zero_grad()
            preds=Encoder_z.forward(fake_images.detach())
            curr_loss=loss_function(preds,curr_noise)
            curr_loss.backward()
            optimizer.step()
            loss+=curr_loss

        time_taken=time.time()-start_time
        print("Time Taken: {}s Epoch no: {} Loss: {}".format(round(time_taken),i,loss))
        if i%save_freq==0 or i==(start_epochs+epochs-1):
            save.save_encoder_checkpoints(Encoder_z,i,optimizer=optimizer)
            print("saving {}th checkpoint".format(i))

if __name__ == '__main__':
    args = options().parser
    Encoder_z=Encoder_z(args.l)
    Generator=Generator(args.ny,args.l)
    train(Generator,Encoder_z,continue_train=args.continue_train,epochs=args.epochs,batch_size=args.batch_size,
          n_batches=args.n_batches,l=args.l,save_freq=args.save_freq,ny=args.ny)




