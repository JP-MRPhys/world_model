import numpy as np
from collections import namedtuple

TIME_FACTOR = 0
NOISE_BIAS = 0
OUTPUT_NOISE = [False, False]
OUTPUT_SIZE = 2


def activations_old(a):
  a = np.tanh(a)
  a[1] = (a[1] + 1) / 2
  #a[2] = (a[2] + 1) / 2
  return a

def activations(a):
  a = np.tanh(a)
  a[0] = (a[0] + 1) / 2
  a[1] = (a[1] + 1) / 2

  bins=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]

  a[0] = np.searchsorted(bins, a[0])/100   #centre fraction to be select between 1 to 10%
  a[1]=np.searchsorted(bins,a[1])          # acceleration factor between 1 to 10

  return a

class Controller():
    def __init__(self):
        self.time_factor = TIME_FACTOR
        self.noise_bias = NOISE_BIAS
        self.output_noise=OUTPUT_NOISE
        self.activations=activations
        self.output_size = OUTPUT_SIZE