### World Model implementation for fastMRI datasets to accelerate MRI scanning 

###  Generative model
##### 1. Train VAE with random actions (need to select acceleration factor and centre fraction), save model as json format (to load multiple times) 
##### 2. Check VAE recons: they are blurry (WIP need to optimize)
##### 3. Generate series with random actions "a" and recons datasets to obtain latent representation "z" required to train RNN


### MDN-RNN 

#####
#####
#####


### Controller 

##### Train linear controller using es 
python train_controller.py fastmri -e 1 -n 4 -t 1 --max_length 100

### Inference using controller to generate new actions


###### Datasets FastMRI https://fastmri.org/ with a custom enviroment, to provide rewards are MSE between gold standard and reconstructed




###### References: https://github.com/hardmaru/WorldModelsExperiments

###### License: MIT Open source


