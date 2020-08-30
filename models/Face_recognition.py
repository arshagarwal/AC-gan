"""
    Author:Bhuvan
    1.Train the model on an appropriate facial recognition dataset
    2.Save the checkpoints for future use.
"""
import tensorflow as tf
from tensorflow import keras

import numpy as np
from tensorflow.keras.models import load_model;
class FR_model(tf.keras.Model):

    def __init__(self):
        super(FR_model,self).__init__();
        """
        Layer Declaration
        Uses one of the following pre-trained network (VGG-FACE | VGG-16 | VGG-19); add only one of these for now
        """
        self.model = load_model('Facenet/facenet_keras.h5')
        self.model.load_weights('Facenet/facenet_keras_weights.h5')
    def __call__(self,x):
        """
        :param x: Images of the shape [batch_size,128,128,3]
        :param l: No of units in the last dense layer
        :return:  Embedding of shape [batch_size,l]
        """
        return self.model(x)


    def train(self,x,y,epochs=10,batch_size=20):
        """
        :param x: Images of the shape  [batch_size,128,128,3]
        :param y: Ground Truth vector of shape [batch_size,len of each ground truth vector]
        :return: void
        """
