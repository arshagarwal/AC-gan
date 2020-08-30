"""
Author: Arsh Agarwal

"""
import tensorflow as tf
import numpy as np

class Discriminator(tf.keras.Model):

    def __init__(self,ny=10):
        """
        :param ny: number of categories Image can be classified into 
        """
        super(Discriminator,self).__init__();
        self.ny=ny;

        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.batch_norm4 = tf.keras.layers.BatchNormalization()
        self.batch_norm5 = tf.keras.layers.BatchNormalization()

        self.relu1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.relu2 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.relu3 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.relu4 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.relu5 = tf.keras.layers.LeakyReLU(alpha=0.2)

        self.conv1=tf.keras.layers.Conv2D(64, 4, (2,2), use_bias=False,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.02), padding='same')
        self.conv2 = tf.keras.layers.Conv2D(128, 4, (2, 2),use_bias=False,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.02), padding='same')
        self.conv3 = tf.keras.layers.Conv2D(256, 4, (2, 2),use_bias=False,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.02), padding='same')
        self.conv4 = tf.keras.layers.Conv2D(512, 4, (2, 2),use_bias=False,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.02), padding='same')
        self.conv5 = tf.keras.layers.Conv2D(1024, 4, (2, 2),use_bias=False,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0,stddev=0.02), padding='same')



        self.flatten=tf.keras.layers.Flatten()
        self.dense1 = tf.keras.layers.Dense(1,activation='sigmoid',use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))
        self.dense2 = tf.keras.layers.Dense(self.ny,activation='softmax', use_bias=False, kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02))




    def call(self,z,training=False):
        """
        Performs both real fake discrimination and categorical classification
        :param z:  Images of the shape [batch_size,128,128,3]
        :return:
        """
        x=self.conv1(z)
        x=self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batch_norm2(x, training)


        x = self.conv3(x)
        x = self.relu3(x)
        x = self.batch_norm3(x, training)


        x = self.conv4(x)
        x = self.relu4(x)
        x = self.batch_norm4(x, training)


        x = self.conv5(x)
        x=self.relu5(x)
        x = self.batch_norm5(x, training)


        x=self.flatten(x)

        o1=self.dense1(x)
        o2=self.dense2(x)
        return o1 ,o2

class Generator(tf.keras.Model):
    """Given latent vector, produces Image of the shape (128,128,3)
      latent vector shape: [batch_size,z+ny]
      where
      z: is the number of parameters into which which image is encoded using Encoder_z (100)
      ny:number of attributes in an image
      """
    def __init__(self,ny,l=100):
        """

        :param l: number of units in the latent vector
        :param ny: number of categories
        """
        super(Generator,self).__init__();
        self.l=l

        self.dense=tf.keras.layers.Dense(4*4*1024,use_bias=False,kernel_initializer=tf.keras.initializers.RandomNormal(0.,0.02))
        self.reshape = tf.keras.layers.Reshape((4, 4, 1024))

        # embedding layer
        self.embedding =tf.keras.layers.Embedding(ny,20,embeddings_initializer=tf.keras.initializers.RandomNormal(0,0.02))
        self.dense2=tf.keras.layers.Dense(4*4*1,use_bias=False,kernel_initializer=tf.keras.initializers.RandomNormal(0,0.02))
        self.reshape2=tf.keras.layers.Reshape((4,4,1))



        # conv layers
        self.conv0 = tf.keras.layers.Conv2DTranspose(512, 4, (2,2), padding='same', use_bias=False,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0,stddev=0.02))
        self.conv1 = tf.keras.layers.Conv2DTranspose(256, 4, (2,2), padding='same', use_bias=False,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0,stddev=0.02))
        self.conv2 = tf.keras.layers.Conv2DTranspose(128, 4, (2,2), padding='same', use_bias=False,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0,stddev=0.02))
        self.conv3 = tf.keras.layers.Conv2DTranspose(64, 4, (2,2), padding='same', use_bias=False,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0,stddev=0.02))
        self.conv4 = tf.keras.layers.Conv2DTranspose(3, 4, (2,2), padding='same', activation='tanh', use_bias=False,kernel_initializer=tf.keras.initializers.RandomNormal(mean=0,stddev=0.02))

        # batch_norm layers
        self.batch_norm0=tf.keras.layers.BatchNormalization()
        self.batch_norm1=tf.keras.layers.BatchNormalization()
        self.batch_norm2 = tf.keras.layers.BatchNormalization()
        self.batch_norm3 = tf.keras.layers.BatchNormalization()
        self.dense_batch_norm=tf.keras.layers.BatchNormalization()

        # leaky relu layers
        self.relu1 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.relu2 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.relu3 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.relu4 = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.relu5 = tf.keras.layers.LeakyReLU(alpha=0.2)

        #concat layer
        self.concat=tf.keras.layers.concatenate



    def call(self,input,labels,training=False):
        """
        Generates synthetic images of the shape [batch_size,128,128,3]
        :param input: Latent vector + attributes of the shape [batch_size,z+l]
        :return: synthetic images of the shape [batch_size,128,128,3]
        """
        # creating the embedding
        embedding=self.embedding(labels)
        embedding=self.dense2(embedding)
        assert embedding.shape == (input.shape[0],16), " check generator embedding shape {} ".format(embedding.shape)
        embedding=self.reshape2(embedding)

        x=self.dense(input)
        x=self.relu1(x)
        x = self.dense_batch_norm(x, training=training)
        x=self.reshape(x)

        # concatenating embedding
        x=self.concat([x,embedding],axis=-1)

        assert x.shape == (input.shape[0],4,4,1025), "check embedding concatenation"

        x = self.conv0(x)
        x = self.relu2(x)
        x = self.batch_norm0(x, training=training)

        x=self.conv1(x)
        x = self.relu3(x)
        x = self.batch_norm1(x, training=training)

        x = self.conv2(x)
        x = self.relu4(x)
        x = self.batch_norm2(x, training=training)

        x = self.conv3(x)
        x = self.relu5(x)
        x = self.batch_norm3(x, training=training)

        x = self.conv4(x)

        return x
    def get_label(self,y):
        """

        :param y: categories of y for the input of the shape [batch_size,ny]
        :return: vector of shape [batch_size,] showing labels for each sample
        """
        batch_size=y.shape[0]
        labels=np.zeros(shape=(y.shape[0],))
        for i in range(batch_size):
            curr_max=max(y[i]).numpy()
            for j in range(y.shape[1]):
                if y[i][j]==curr_max:
                    labels[i]=j;
                    break;
        return tf.convert_to_tensor(labels)











