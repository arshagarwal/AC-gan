import argparse

class options:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--data_path",required=True,help="String that gives the path to the Images")
        parser.add_argument("--epochs",type=int,default=50,help="number of epochs")
        parser.add_argument("--batch_size",type=int,default=20,help="Batch size to be used")
        parser.add_argument("--lambda_A",type=float,default=2.0,help="factor by which discriminator loss is divided")
        parser.add_argument("--lambda_B",type=float,default=1.0,help=" factor by which attribute classification loss is multiplied w.r.t real/fake loss.")
        parser.add_argument("--loss_c",default="cce",help="loss function used for classification")
        parser.add_argument("--mode",default="generation",help="specifies the mode of training as either 'generation' or 'augmentation'")
        parser.add_argument("--l",type=int,default=100,help="number of units in noise vector")
        parser.add_argument("--save_freq",type=int,default=10,help='save epoch frequency')
        parser.add_argument("--continue_train",action="store_true",help="flag to indicate continue train")
        parser.add_argument("--max_save",default=5,help="denotes number of checkpoints to be saved ")
        self.parser=parser.parse_args()
