### World Model implementation for fastMRI datasets: towards autonoumus imaging (selecting acceleration factors)

###  Generative model
 1. Train VAE with random actions (select acceleration factor and centre fraction), save model as json format (to load multiple times) 
 2. Check VAE recons: they will be blurry use autoregressive flows (VQ-VAE) in future 
 3. Generate series datasets with random actions "aceeleration and centre fraction" reconstructed images to obtain latent representation "z" required to train RNN


### MDN-RNN 

 1. Train MDN-RNN with series datasets i.e. actions and latent vae-representation rnn.train() to obtain probablity distribution over action given latents 
 2. Save the model in json format
 3. Commit the models to a git-repo


### Controller 

 1. Clone git-repo in a GCP VM-instance with 64 cores to train the controller
 2. Install MPI (https://www.youtube.com/watch?v=FOqhiX4X5xw), anaconda, and mpi4py (via pip)
 3. Train linear controller using CMA-ES: No tensorflow required, using David Ha's implementation for details references below

python train_controller.py fastmri -e 1 -n 4 -t 1 --max_length 100

Inference using controller to generate new actions (see the Jupyter notebook)


![alt text](https://github.com/JP-MRPhys/world_model/blob/master/models/trained_models/CVAE/images_1/_rollout_12a_8.0_.png)

e.g. VAE versus traditional recons at 8X acceleration acceleration factors obtained via controller

###### Datasets: Knee MRI:FastMRI https://fastmri.org/ with a custom enviroment, to provide rewards are MSE between gold standard and reconstructed images

###### References: https://github.com/hardmaru/WorldModelsExperiments

###### License: MIT Open source

##### Acknowledgements: Google Cloud Platform for research credits 


