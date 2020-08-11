### World Model implementation for fastMRI datasets towards autonoumus imaging 

###  Generative model
 1. Train VAE with random actions (need to select acceleration factor and centre fraction), save model as json format (to load multiple times) 
 2. Check VAE recons: they will be blurry use autoregressive flows 
 3. Generate series datasets with random actions "a" and relavant, reconstructed images to obtain latent representation "z" required to train RNN


### MDN-RNN 

 1. Train MDN-RNN with series datasets i.e. actions and latent vae-representation rnn.train()
 2. Save the model in json format
 3. Commit the models to a git-repo


### Controller 

 1. Clone git-repo in a GCP VM-instance with 64 cores to train the controller
 2. Install MPI (https://www.youtube.com/watch?v=FOqhiX4X5xw), anaconda, and mpi4py (via pip)
 3. Train linear controller using CMA-ES: No tensorflow required, using David Ha's implementation for details references below

python train_controller.py fastmri -e 1 -n 4 -t 1 --max_length 100

### Inference using controller to generate new actions

![alt text](https://github.com/JP-MRPhys/world_model/blob/master/models/trained_models/CVAE/images_1/_rollout_12a_8.0_.png)




###### Datasets FastMRI https://fastmri.org/ with a custom enviroment, to provide rewards are MSE between gold standard and reconstructed images




###### References: https://github.com/hardmaru/WorldModelsExperiments

###### License: MIT Open source

##### Acknowledgements: Google Cloud Platform for research credits 


