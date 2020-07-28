### World Model implementation for fastMRI datasets to accelerate MRI scanning 

###  Generative model
##### 1. Train VAE with random actions (need to select acceleration factor and centre fraction), save model as json format (to load multiple times) 
##### 2. Check VAE recons: they are blurry (WIP need to optimize)
##### 3. Generate series with random actions "a" and relavant, reconstructed images to obtain latent representation "z" required to train RNN


### MDN-RNN 

##### 1. Train MDN-RNN with series datasets i.e. actions and latent representation rnn.train()
##### 2. Save the model in json format, so can create replicas 
##### 3. Model the models to a git-repo


### Controller 

##### Clone git-repo in a GCP VM-instance with 64 cores to train the controller
##### Train linear controller using CMA-ES: No tensorflow required, see David Ha's implementation for details
python train_controller.py fastmri -e 1 -n 4 -t 1 --max_length 100

### Inference using controller to generate new actions


###### Datasets FastMRI https://fastmri.org/ with a custom enviroment, to provide rewards are MSE between gold standard and reconstructed images




###### References: https://github.com/hardmaru/WorldModelsExperiments

###### License: MIT Open source


