
"""
author: Arsh Agarwal
Code to declare and train the encoders.
"""
import tensorflow as tf
class Encoder_z(tf.keras.Model):
    def __init__(self,l=100):
        """
        :param l: length of latent vector to be obtained
        """
        super(Encoder_z,self).__init__();
        self.l=l
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.batch_norm4 = tf.keras.layers.BatchNormalization()
        self.batch_norm5 = tf.keras.layers.BatchNormalization()
        self.conv_1 = tf.keras.layers.Conv2D(filters=32,kernel_size=5,strides=(2,2),activation='relu')
        self.conv_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=(2, 2), activation='relu')
        self.conv_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=(2, 2), activation='relu')
        self.conv_4 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=(2, 2), activation='relu')
        self.conv_5 = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=(2, 2), activation='relu')
        self.flatten=tf.keras.layers.Flatten()
        self.dense1=tf.keras.layers.Dense(4096,'relu')
        self.dense2 = tf.keras.layers.Dense(l)

    def call(self,inputs,training=False):
        """
        Gives the encoded output
        :param inputs: Images of the shape [batch_size,128,128,3]
        :param training: Boolean to start or stop training
        :return: the encoded latent vector z
        """
        x = self.conv_1(inputs)
        x = self.batch_norm1(x)
        #second layer
        x = self.conv_2(x)
        x = self.batch_norm2(x)
        #3rd layer
        x = self.conv_3(x)
        x = self.batch_norm3(x)
        #4th layer
        x = self.conv_4(x)
        x = self.batch_norm4(x)
        #5th layer
        x = self.conv_5(x)
        x = self.batch_norm5(x)
        x = self.flatten(x)
        #z vector a.k.a Encoded image vector
        z = self.dense1(x)
        z = self.dense2(z)
        return z

class Encoder_y(tf.keras.Model):

    def __init__(self,ny):
        """

        :param ny: the length of the attribute vector
        """
        super(Encoder_y,self).__init__();
        self.batch_norm1 = tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.batch_norm4 = tf.keras.layers.BatchNormalization()
        self.batch_norm5 = tf.keras.layers.BatchNormalization()
        self.conv_1 = tf.keras.layers.Conv2D(filters=32, kernel_size=5, strides=(2, 2), activation='relu')
        self.conv_2 = tf.keras.layers.Conv2D(filters=64, kernel_size=5, strides=(2, 2), activation='relu')
        self.conv_3 = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=(2, 2), activation='relu')
        self.conv_4 = tf.keras.layers.Conv2D(filters=256, kernel_size=5, strides=(2, 2), activation='relu')
        self.conv_5 = tf.keras.layers.Conv2D(filters=128, kernel_size=5, strides=(2, 2), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(512, 'relu')
        self.dense2 = tf.keras.layers.Dense(ny)

    def call(self, inputs, training=False):
        """
        Gives encoded attribute vector as attribute.
        :param inputs: Images of the shape [batch_size,128,128,3]
        :param training:
        :return: return the attribute vector of length self.ny
        """
        x = self.conv_1(inputs)
        x = self.batch_norm1(x)
        # second layer
        x = self.conv_2(x)
        x = self.batch_norm2(x)
        # 3rd layer
        x = self.conv_3(x)
        x = self.batch_norm3(x)
        # 4th layer
        x = self.conv_4(x)
        x = self.batch_norm4(x)
        # 5th layer
        x = self.conv_5(x)
        x = self.batch_norm5(x)
        x = self.flatten(x)

        y = self.dense1(x)
        y = self.dense2(y)
        return y