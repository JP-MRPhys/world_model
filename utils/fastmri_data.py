import h5py
import numpy as np
import pathlib
from utils.subsample import MaskFunc
import utils.transforms as T

# a number of utility functions to read and transform the data


def get_training_pair(file, centre_fraction, acceleration):

    """


    :param file: The training image
    :param centre_fraction: randomly generated centre fraction
    :param acceleration: randomly generated
    :return:
    """


    hf = h5py.File(file)

    volume_kspace = hf['kspace'][()]
    volume_image = hf['reconstruction_esc'][()]
    mask_func = MaskFunc(center_fractions=[centre_fraction], accelerations=[acceleration])  # Create the mask function object

    volume_kspace_tensor = T.to_tensor(volume_kspace)
    masked_kspace, mask = T.apply_mask(volume_kspace_tensor, mask_func)
    masked_kspace_np=masked_kspace.numpy().reshape(masked_kspace.shape)

    return np.expand_dims(volume_image,3), masked_kspace_np


def get_training_pair_images(file, centre_fraction, acceleration):

    """

    :param file: The training image
    :param centre_fraction: randomly generated centre fraction
    :param acceleration: randomly generated
    :return: true gold standard and the fft of the masked k-space image

    """
    hf = h5py.File(file)
    volume_kspace = hf['kspace'][()]
    volume_image = hf['reconstruction_esc'][()]
    mask_func = MaskFunc(center_fractions=[centre_fraction], accelerations=[acceleration])  # Create the mask function object

    volume_kspace_tensor = T.to_tensor(volume_kspace)
    masked_kspace, mask =  T.apply_mask(volume_kspace_tensor, mask_func)
    ##masked_kspace_np=masked_kspace.numpy().reshape(masked_kspace.shape)
    
    recon_image= T.ifft2(masked_kspace) 			 # complex image
    recon_image_abs= T.complex_abs(masked_kspace)                # compute absolute value to get a real image
    #recon_image_rss= T.root_sum_of_square(recon_image_abs,dim=0) # compute absolute rss

    return np.expand_dims(volume_image,3), recon_image_abs.numpy()


def get_training_pair_images_vae(file, centre_fraction, acceleration, image_size=256):
    """
    :param file: The training image
    :param centre_fraction: randomly generated centre fraction
    :param acceleration: randomly generated
    :return: true gold standard and the fft of the masked k-space image

    """
    hf = h5py.File(file)
    volume_kspace = hf['kspace'][()]
    volume_image = hf['reconstruction_esc'][()]
    mask_func = MaskFunc(center_fractions=[centre_fraction],
                         accelerations=[acceleration])  # Create the mask function object

    volume_kspace_tensor = T.to_tensor(volume_kspace)
    masked_kspace, mask = T.apply_mask(volume_kspace_tensor, mask_func)
    ##masked_kspace_np=masked_kspace.numpy().reshape(masked_kspace.shape)

    volume_image = T.center_crop(volume_image, shape=[image_size, image_size])

    recon_image = T.ifft2(masked_kspace)  # complex image
    recon_image = T.complex_center_crop(recon_image, shape=[volume_image.shape[1], volume_image.shape[2]])

    volume_image, mean, std = T.normalize_instance(volume_image)
    # recon_image=T.normalize_instance(recon_image)

    recon_image_abs = T.complex_abs(recon_image)  # compute absolute value to get a real image
    # recon_image_rss= T.root_sum_of_square(recon_image_abs,dim=0) # compute absolute rss

    recon_image_abs, mean, std = T.normalize_instance(recon_image_abs)

    return np.expand_dims(volume_image, 3), recon_image_abs.numpy()


def get_random_accelerations(high):
    """
       : we apply these to fully sampled k-space to obtain q
       :return:random centre_fractions between 0.1 and 0.001 and accelerations between 1 and low
    """
    acceleration = np.random.randint(1, high=high, size=1)
    centre_fraction = np.random.uniform(0, 1, 1)
    decimal = np.random.randint(1, high=3, size=1)
    centre_fraction = centre_fraction / (10 ** decimal)

    return float(centre_fraction), float(acceleration)


def get_random_accelerations_old():

   """
      : we apply these to fully sampled k-space to obtain q
      :return:random centre_fractions between 0.1 and 0.001 and accelerations between 1 and 15
   """
   acceleration = np.random.randint(1, high=15, size=1)
   centre_fraction = np.random.uniform(0, 1, 1)
   decimal = np.random.randint(1, high=3, size=1)
   centre_fraction = centre_fraction / (10 ** decimal)

   return float(centre_fraction), float(acceleration)



def train(datadir):
    files = list(pathlib.Path(datadir).iterdir())
    np.random.shuffle(files)
    for file in files:
        print(file)
        centre_fraction,acceleration=get_random_accelerations(high=5)
        image, masked_kspace =get_training_pair_images(file, centre_fraction=centre_fraction,acceleration=acceleration)
        print(image.shape)
        print(masked_kspace.shape)


if __name__ == '__main__':

    #training_datadir='/media/jehill/Data/ML_data/fastmri/singlecoil/train/singlecoil_train/'
    training_datadir='/media/jehill/DATA/ML_data/fastmri/singlecoil/train/singlecoil_train/'
    train(training_datadir)