import pandas as pd
import numpy as np
import pathlib
from datetime import datetime, timedelta
from utils.fastmri_data import get_training_pair_images_vae, get_random_accelerations
import random
#action is the random accelerating factors  see how these are selected by others
# state after taking action is the undersampled image
# reward is the squared difference between undersample and fully sampled gold standard

class fastMRIEnviroment(object):
    def __init__(self, datadir):

        self.training_datadir=datadir

        self.filenames = list(pathlib.Path(self.training_datadir).iterdir())
        self.number_files=len(self.filenames)
        print("Number training data " + str(self.number_files))
        self.counter=1
        self.done=False
        self.batch_size=3

    def get_reward(self, training_images, training_labels):

        mse = (np.square(training_images - training_labels)).mean(axis=None)

        return mse

    def get_action(self):
        centre_fraction, acceleration=get_random_accelerations(high=10)
        return centre_fraction, acceleration


    def step(self, action):

        # action is the selection acceleration factor
        #centre_fraction, acceleration=self.get_action()


        centre_fraction, acceleration = action[0], action[1]
        training_images, training_labels =  get_training_pair_images_vae(self.filenames[self.counter], centre_fraction, acceleration)

        reward=self.get_reward(training_images[:,:,:,0], training_labels)
        print( self.counter % self.batch_size)

        if (self.counter % self.batch_size == 0):
            self.done=True
            self.counter=1
            random.shuffle(self.filenames)
            print("Episode complete")

        else:
            self.done=False
            self.counter +=1

        # state, reward, done,
        return training_labels, training_images, action, reward, self.done

    def get_filenames(self):

        filenames=list(pathlib.Path(self.training_datadir).iterdir())
        np.random.shuffle(filenames)
        print("Number training data " + str(len(filenames)))


    def reset(self):

        self.done = True
        self.counter = 1
        random.shuffle(self.filenames)


