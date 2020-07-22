import pandas as pd
import numpy as np
import pathlib
from datetime import datetime, timedelta
from fastmri_data import get_training_pair_images_vae, get_random_accelerations

import random

#action is the random accelerating factors  see how these are selected by others
#state after taking action is the undersampled image
#reward is the squared difference between undersample and fully sampled gold standard

class fastMRIEnviroment(object):
    def __init__(self, datadir):

        self.training_datadir=datadir

        self.filenames = list(pathlib.Path(self.training_datadir).iterdir())
        self.number_files=len(self.filenames)
        print("Number training data " + str(self.number_files))
        self.counter=1
        self.done=False
        self.batch_size=1
        random.shuffle(self.filenames)

    def get_reward(self, training_images, training_labels):

        mse = (np.square(training_images - training_labels)).mean(axis=None)

        return mse

    def get_action(self):
        centre_fraction, acceleration=self.get_random_accelerations(high=10)
        return centre_fraction, acceleration

    def get_random_accelerations(self, high):
        """
           : we apply these to fully sampled k-space to obtain q
           :return:random centre_fractions between 0.1 and 0.001 and accelerations between 1 and low
        """
        acceleration = np.random.randint(1, high=high, size=1)
        centre_fraction = np.random.uniform(0, 1, 1)
        decimal = np.random.randint(1, high=3, size=1)
        centre_fraction = centre_fraction / (10 ** decimal)

        return float(centre_fraction), float(acceleration)


    def step(self, action):

        # action is the selection acceleration factor
        #centre_fraction, acceleration=self.get_action()


        centre_fraction, acceleration = action[0], action[1]
        random.shuffle(self.filenames)
        randomIndex=random.randint(0, high=30,size=1)
        file=self.filenames[randomIndex]
        print(file)
        training_images, training_labels =  get_training_pair_images_vae(file, centre_fraction, acceleration)

        reward=self.get_reward(training_images[:,:,:,0], training_labels[:,:,:,0])
        print( self.counter % self.batch_size)

        if (self.counter % self.batch_size == 0):
            self.done=True
            self.counter=1
            random.shuffle(self.filenames)
            print("Episode complete")

        else:
            self.done=False
            self.counter +=1

        random.shuffle(self.filenames)

        # state, reward, done,
        return training_labels[:,:,:,0], reward, self.done

    def get_filenames(self):

        filenames=list(pathlib.Path(self.training_datadir).iterdir())
        np.random.shuffle(filenames)
        print("Number training data " + str(len(filenames)))


    def reset(self):

        self.done = True
        self.counter = 1
        random.shuffle(self.filenames)



    def close(self):
      pass


    def seed(self, seed):
        random.seed(seed)
        np.random.seed

        # Not really sure if this will work for reproducbility, add it for make the programms run


if __name__ == '__main__':

    env=fastMRIEnviroment(datadir='/media/DATA/ML_data/fastmri/singlecoil/train/singlecoil_train/')


    while True:
        action = np.ones(2)
        action[0], action[1] = env.get_action()

        obs,reward, done= env.step(action)
        print(np.shape(obs))
        print(reward)
        if done:
            print("New file")