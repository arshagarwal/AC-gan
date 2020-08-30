
"""
author: Bhuvan Sachdeva
Code to use bmi predictor
"""
import tensorflow as tf
class BMI(tf.keras.model):
    def __init__(self, num):
        self.conv_1 = tf.keras.layers.Conv2D(filters=32,kernel_size=5,strides=(2,2),activation='relu')
        self.conv_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=(2, 2), activation='relu')
        self.conv_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=(2, 2), activation='relu')
        self.conv_4 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=(2, 2), activation='relu')
        self.conv_5 = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=(2, 2), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(4096,'relu')
        self.dense2 = tf.keras.layers.Dense(num)

    def __call__(self, x):
        
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.dense1(x)
        x = self.dense2(x)
        
        return x

    def train(self, x, y):

        pass