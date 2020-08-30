"""
Command Line options for test_cgan
Author: Arsh Agarwal
"""
import argparse
class options:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('n_classes', type=int, default=2, help="number of classes in the data")
        parser.add_argument('--n_samples',type=int,default=50,help="Number of images to be produced needed only in generation mode")
        parser.add_argument('--l',type=int,default=100,help="number of units in the latent vector")
        parser.add_argument('--z_path',default="",help='path to z tensors, needed for augmentation mode')
        parser.add_argument('--y_path',default="",help='path to corresponding y tensors, needed in augmentation mode')
        parser.add_argument('--mode',default="generation",help='String to specify the mode')
        self.parser=parser.parse_args()