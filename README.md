### World Model implementation for fastMRI datasets to accelerate MRI scanning 

###  Generative model
#### 1. Train VAE with random actions (need to select acceleration factor and centre fraction), save model as json format (to load multiple times) 
#### 2. Check VAE recons: they are blurry (WIP need to optimize)
#### 3. Generate series with random actions and k-spaces datasets to train probablistics RNN


### MDN-RNN 

####
####
####

### Train controller 


### Inference using controller to generate new actions


##### Datasets FastMRI https://fastmri.org/ with a custom enviroment, to provide rewards are MSE between gold standard and reconstructed
