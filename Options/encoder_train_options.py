import argparse

class options:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--n_batches",type=int,default=10,help="String that gives the path to the Images")
        parser.add_argument("--epochs",type=int,default=50,help="number of epochs")
        parser.add_argument("--batch_size",type=int,default=20,help="Batch size to be used")
        parser.add_argument("--l",type=int,default=100,help="number of units in noise vector")
        parser.add_argument("--save_freq",type=int,default=10,help='save epoch frequency')
        parser.add_argument("--continue_train",action="store_true",help="flag to indicate continue train")
        parser.add_argument("--ny",default=2,help="number of labels ")

        self.parser=parser.parse_args()
