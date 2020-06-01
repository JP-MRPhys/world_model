#import tensorflow as tf
import tensorflow.compat.v1 as tf  #NOTE: To train on tensorflow version 2.0
tf.disable_v2_behavior()

import h5py
import os
import numpy as np
import pathlib
#from utils.subsample import MaskFunc
#import utils.transforms as T
from matplotlib import pyplot as plt
from utils.fastmri_data import get_training_pair_images_vae, get_random_accelerations
import math
import logging
import shutil

LOG_FILENAME="./logs/VAE_TRAINING.LOG"
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

MAX_ROLLOUTS=2000
SERIES_DIR = "./DATA"

if  not os.path.exists(SERIES_DIR):
    os.makedirs(SERIES_DIR)

LATENT_DIM=64

class CVAE(tf.keras.Model):
    def __init__(self):

        super(CVAE, self).__init__()

        #TODO: add config parser

        self.training_datadir='/media/DATA/ML_data/fastmri/singlecoil/train/singlecoil_train/'

        self.BATCH_SIZE = 16
        self.num_epochs = 25
        self.learning_rate = 1e-3
        self.model_name="CVAE"

        self.image_dim = 128
        self.channels = 1
        self.latent_dim = LATENT_DIM

        self.kernel_size = 3
        lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.3)
        self.activation = lrelu

        self.input_image_1 = tf.placeholder(tf.float32, shape=[None, 256, 256, self.channels]) #for time being resize images
        self.input_image = tf.image.resize_images(self.input_image_1, [np.int(self.image_dim), np.int(self.image_dim)])

        self.gold_standard_image_1 = tf.placeholder(tf.float32,
                                            shape=[None, 256, 256, self.channels])  # for time being resize images
        self.gold_standard_image = tf.image.resize_images(self.gold_standard_image_1, [np.int(self.image_dim), np.int(self.image_dim)])

        self.image_shape = self.input_image.shape[1:]
        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')

        self.encoder = self.inference_net()
        self.decoder = self.generative_net()  # note these are keras model

        self.mean, self.logvar = tf.split(self.encoder(self.input_image), num_or_size_splits=2, axis=1)
        self.z = self.reparameterize(self.mean, self.logvar)
        logits = self.decoder(self.z)
        self.reconstructed = tf.sigmoid(logits)



        # calculate the KL loss
        var = tf.exp(self.logvar)
        kl_loss = 0.5 * tf.reduce_sum(tf.square(self.mean) + var - 1. - self.logvar)

        # cal mse loss
        self.sse_loss = 0.5 * tf.reduce_sum(tf.square(self.gold_standard_image - logits))
        self.total_loss = tf.reduce_mean(kl_loss + self.sse_loss) / self.BATCH_SIZE
        self.list_gradients = self.encoder.trainable_variables + self.decoder.trainable_variables
        self.Optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.total_loss, var_list=self.list_gradients)



        # summary and writer for tensorboard visulization

        tf.summary.image("Reconstructed_image", self.reconstructed)
        tf.summary.image("Input_image", self.input_image)


        tf.summary.scalar("KL", kl_loss)
        tf.summary.scalar("SSE",self.sse_loss)
        tf.summary.scalar("Totalloss", self.total_loss)

        self.merged_summary = tf.summary.merge_all()
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()


        self.logdir = './trained_models/' + self.model_name  # if not exist create logdir
        self.image_dir = self.logdir + '/images/'
        self.model_dir= self.logdir + '/intraining'
        self.final_model_dir = self.logdir + '/final_model'



        #self.gpu_list=['/gpu:0', '/gpu:1' ,'/gpu:2', '/gpu:3']
        self.gpu_list = ['/gpu:0']

        print("Completed creating the model")
        logging.debug("Completed creating the model")

        if (os.path.exists(self.image_dir)):
            shutil.rmtree(self.image_dir, ignore_errors=True)
            os.makedirs(self.image_dir)
        else:
            os.makedirs(self.image_dir)


    def inference_net(self):
        input_image = tf.keras.layers.Input(self.image_shape)  # 224,224,1
        net = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=(2, 2), activation='relu')(input_image)  # 112,112,32
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Conv2D(
            filters=64, kernel_size=3, strides=(2, 2), activation='relu')(net)  # 56,56,64
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Conv2D(
            filters=32, kernel_size=3, strides=(2, 2), activation='relu')(net)  # 56,56,64
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Flatten()(net)
        # No activation
        net = tf.keras.layers.Dense(self.latent_dim + self.latent_dim)(net)
        net = tf.keras.Model(inputs=input_image, outputs=net)

        return net

    def generative_net(self):
        latent_input = tf.keras.layers.Input((self.latent_dim,))
        net = tf.keras.layers.Dense(units=8 * 8 * 128, activation=tf.nn.relu)(latent_input)
        net = tf.keras.layers.Reshape(target_shape=(8, 8, 128))(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Conv2DTranspose(
            filters=256,
            kernel_size=5,
            strides=(2, 2),
            padding="SAME",
            activation=self.activation)(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Conv2DTranspose(
            filters=128,
            kernel_size=5,
            strides=(2, 2),
            padding="SAME",
            activation=self.activation)(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Conv2DTranspose(
            filters=64,
            kernel_size=5,
            strides=(2, 2),
            padding="SAME",
            activation=self.activation)(net)
        net = tf.keras.layers.BatchNormalization()(net)
        net = tf.keras.layers.Conv2DTranspose(
            filters=32,
            kernel_size=5,
            strides=(2, 2),
            padding="SAME",
            activation=self.activation)(net)
        net = tf.keras.layers.BatchNormalization()(net)
        # No activation
        net = tf.keras.layers.Conv2DTranspose(
            filters=self.channels, kernel_size=3, strides=(1, 1), padding="SAME", activation=None)(net)
        upsampling_net = tf.keras.Model(inputs=latent_input, outputs=net)
        return upsampling_net

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(tf.shape(mean))
        # return eps * tf.exp(logvar * .5) + mean
        return eps * tf.sqrt(tf.exp(logvar)) + mean


    def encoder_predict(self, input_image):

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as self.sess:

            a = os.path.join(self.final_model_dir, self.model_name)
            self.saver.restore(self.sess, a)

            feed_dict = {self.input_image_1: input_image}

            z, mean, logvar= self.sess.run(
                [self.z, self.mean, self.logvar],
                feed_dict=feed_dict)

            return z, mean, logvar

    def train(self):

        for d in self.gpu_list:

         with tf.device(d):
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as self.sess:

                #learning_rate=1e-3
                counter = 0


                self.train_writer = tf.summary.FileWriter(self.logdir, tf.get_default_graph())
                self.sess.run(self.init)

                # so can see improvement fix z_samples
                z_samples = np.random.uniform(-1, 1, size=(self.BATCH_SIZE, self.latent_dim)).astype(np.float32)

                for epoch in range(0, self.num_epochs):

                    print("************************ epoch:" + str(epoch) + "*****************")
                    logging.debug("************************ epoch:" + str(epoch) + "*****************")

                    learning_rate=self.step_decay(epoch)

                    filenames = list(pathlib.Path(self.training_datadir).iterdir())
                    np.random.shuffle(filenames)
                    print("Number training data " + str(len(filenames)))
                    np.random.shuffle(filenames)
                    reward=0
                    for file in filenames:

                        centre_fraction, acceleration = get_random_accelerations(high=10)
                        # training_images: fully sampled MRI images
                        # training labels: , obtained using various mask functions, here we obtain using center_fraction =[], acceleration=[]
                        training_images, training_labels = get_training_pair_images_vae(file, centre_fraction, acceleration)
                        [batch_length, x, y, z] = training_images.shape


                        for idx in range(0, batch_length, self.BATCH_SIZE):

                            batch_images = training_images[idx:idx + self.BATCH_SIZE, :, :]
                            batch_labels = training_labels[idx:idx + self.BATCH_SIZE, :, :]


                            feed_dict = {self.input_image_1: batch_images,
                                         self.gold_standard_image_1 : batch_labels,
                                         self.learning_rate: learning_rate}

                            summary, reconstructed_images, opt, loss, recon_loss = self.sess.run( [self.merged_summary, self.reconstructed, self.Optimizer, self.total_loss, self.sse_loss],
                                feed_dict=feed_dict)

                            elbo = -loss

                            reward += recon_loss


                            if math.isnan(elbo):
                                logging.debug("Epoch: " + str(epoch) + "stopping as elbo is nan")
                                print("Training stopped")
                                break


                            #sampled_image = self.sess.run(self.reconstructed, feed_dict={self.z: z_samples})
                            print("Epoch: " + str(epoch) + " learning rate:" + str(learning_rate) + "ELBO: " + str(elbo))



                            counter += 1



                        if (counter % 50 == 0):
                                logging.debug("Epoch: " + str(epoch) + " learning rate:" + str(learning_rate) + "ELBO: " + str(elbo))


                    sampled_image = self.sess.run(self.reconstructed, feed_dict={self.z: z_samples})

                    logging.debug("Epoch: " + str(epoch) + "completed")
                    print("epoch:" + str(epoch) + "Completed")
                    self.save_images(reconstructed_images,"recon"+str(epoch))
                    self.save_images(sampled_image,"sample"+str(epoch))

                    if (epoch % 10 == 0):
                        logging.debug("Epoch: " + str(epoch) + " learning rate:" + str(learning_rate) + "ELBO: " + str(elbo))

                        self.save_model(self.model_dir,  str(epoch))

                    if (epoch % 20 == 0):
                        self.train_writer.add_summary(summary)


                print("Training completed .... Saving model")
                logging.debug(("Training completed .... Saving model"))
                self.save_model(self.final_model_dir, self.model_name)
                print("All completed good bye")

    def sample(self):
        with tf.device('/gpu:0'):
            with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as self.sess:


                a = os.path.join(self.final_model_dir, self.model_name)
                self.saver.restore(self.sess, a)

                # so can see improvement fix z_samples
                z_samples = np.random.uniform(-1, 1, size=(self.batch_size, self.latent_dim)).astype(np.float32)
                sampled_image = self.sess.run(self.reconstructed, feed_dict={self.z: z_samples})

                return sampled_image

    def save_model(self, dir, name):


        a=os.path.join(dir, name)

        print ("Saving the model after training")
        if not (os.path.exists(a)):
            os.makedirs(a)
            print("Create a model dir:" +  a)


        self.saver.save(self.sess, os.path.join(a))
        print("Completed saving the model")
        logging.debug("Completed saving the model")



    def load_model(self):

        print ("Checking for the model")
        a=os.path.join(self.final_model_dir, self.model_name)
        print(a)

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as self.sess:
            self.saver.restore(self.sess, a)
            print ("Session restored")

            #now do the processing before session is close

            return

    def step_decay(self, epoch):
        initial_lrate=0.001
        drop = 0.5
        epochs_drop=2
        lrate= initial_lrate* math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate

    def save_images(self, numpy_array, tag):

        fig = plt.figure(figsize=(4,4))

        for i in range(numpy_array.shape[0]):
            plt.subplot(4,4,i+1)
            plt.imshow(numpy_array[i,:,:,0], cmap='gray')
            plt.axis("off")

        filename=self.image_dir + '_image_at_epoch_' + tag + '_.png';
        plt.savefig(filename)

    def generate_rollouts(self):


        file_list=[]
        action_list=[]
        reward_list=[]
        mu_list= []
        logvar_list =[]



        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as self.sess:

            a = os.path.join(self.final_model_dir, self.model_name)
            self.saver.restore(self.sess, a)

            filenames = list(pathlib.Path(self.training_datadir).iterdir())
            np.random.shuffle(filenames)

            counter=0
            np.random.shuffle(filenames)
            for file in filenames:

                print(file)

                if counter == MAX_ROLLOUTS:
                    break

                reward=0

                centre_fraction, acceleration = get_random_accelerations(high=10)
                # training_images: fully sampled MRI images
                # training labels: , obtained using various mask functions, here we obtain using center_fraction =[], acceleration=[]
                training_images, training_labels = get_training_pair_images_vae(file, centre_fraction, acceleration)
                [batch_length, x, y, z] = training_images.shape

                action=(centre_fraction, acceleration)

                for idx in range(0, batch_length, self.BATCH_SIZE):
                    print(reward)

                    batch_images = training_images[idx:idx + self.BATCH_SIZE, :, :]
                    batch_labels = training_labels[idx:idx + self.BATCH_SIZE, :, :]

                    feed_dict = {self.input_image_1: batch_images,
                                 self.gold_standard_image_1: batch_labels}

                    z, mean, logvar, reconstructed_images, recon_loss,  = self.sess.run(
                        [self.z, self.mean, self.logvar, self.reconstructed,  self.sse_loss],
                        feed_dict=feed_dict)

                    reward += recon_loss

                    counter += 1

                print("Rollout completed: " + str(counter))

                file_list.append(file)
                action_list.append(action)
                reward_list.append(reward)
                mu_list.append(mean)
                logvar_list.append(logvar)

            np.savez_compressed(os.path.join(SERIES_DIR, "vae_series.npz"), datafile=file_list, action=action_list, mu=mu_list,
                                    logvar=logvar_list, reward=reward_list,)


if __name__ == '__main__':

    model=CVAE()
    model.train()
    model.generate_rollouts()