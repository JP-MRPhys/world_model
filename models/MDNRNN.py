import tensorflow as tf
import numpy as np
import shutil
import os
import math
# useful referene for MDN
import json

VAE_LATENT_DIM=64
ACTION_DIM=2
RNN_HIDDEN_UNITS=256
GMM_DIM=5
logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))

Z_factor=1
R_factor=1
LEARNING_RATE=0.001
CLIP_GRADIENT=1

"""
rnn output is only (batch, output_size=[z_dim+1 for reward_dim] and not [batch_size, time_size, output_size) need to check this at later pointer for other envs

Not using teacher forcing to train as one time step is avaliable, so using the true z from fully sampled images or rewards

		rnn_input = np.concatenate([z[:, :-1, :], action[:, :-1, :], rew[:, :-1, :]], axis = 2)
		rnn_output = np.concatenate([z[:, 1:, :], rew[:, 1:, :]], axis = 2) #, done[:, 1:, :]


"""

N=100
SERIES_DIR = "./DATA/"

BATCH_SIZE=100
NUM_STEPS=4000

def get_filelist(N):
    filelist = os.listdir(SERIES_DIR)
    filelist = [x for x in filelist if x != '.DS_Store']
    filelist.sort()
    length_filelist = len(filelist)


    if length_filelist > N:
      filelist = filelist[:N]

    if length_filelist < N:
      N = length_filelist

    return filelist, N


def random_batch(filelist, batch_size=10):
	N_data = len(filelist)
	indices = np.random.permutation(N_data)[0:batch_size]

	z_list = []
	action_list = []
	rew_list = []
	done_list = [] ;z_list_fs = []

	for i in indices:
		#try:
			new_data = np.load(SERIES_DIR + filelist[i], allow_pickle=True)

			mu = new_data['mu']
			log_var = new_data['logvar']
			action = new_data['action']
			reward = new_data['reward'];z_fs = new_data['z_gs']

			#reward = np.expand_dims(reward, axis=2)


			s = log_var.shape

			z = mu + np.exp(log_var/2.0) * np.random.randn(*s)

			z_list.append(z);z_list_fs.append(z_fs)
			action_list.append(action)
			rew_list.append(reward)
		#except:
		#	pass

	z_list = np.array(z_list);z_list_fs=np.array(z_list_fs)
	action_list = np.array(action_list)
	rew_list = np.array(rew_list)

	return z_list, z_list_fs ,action_list, rew_list

def get_rnn_inputs(z, action, reward, z_true):

     # function to format batch data to rnn inputs

       temp=np.zeros((z.shape[0], 3))
       temp[:,0]= action[0]
       temp[:,1]=action[1]
       temp[:,2]=reward

       aa=np.concatenate((z, temp), axis=1)

       aa=np.expand_dims(aa,axis=1)

       #print(aa.shape)
       #print(z_true[1].shape)

       return aa, z_true

filelist, N= get_filelist(N)

class MDNRNN():

    def __init__(self):


        self.Z_dim= VAE_LATENT_DIM
        self.action_dim = ACTION_DIM
        self.gaussian_mixtures_number= GMM_DIM
        self.hidden_units= RNN_HIDDEN_UNITS

        self.model_name='MDNRNN'
        self.logdir = './trained_models/' + self.model_name  # if not exist create logdir
        self.model_dir = self.logdir + '/final_model'


        self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.Z_dim+self.action_dim+1])  # batch size sequence lenght, VAE_z_dim + action_dim+rewards
        self.lstm_inputs_c = tf.placeholder(tf.float32, shape=[None,self.hidden_units])
        self.lstm_inputs_h = tf.placeholder(tf.float32, shape=[None,self.hidden_units])
        self.z_true = tf.placeholder(tf.float32, shape=[None, self.Z_dim])
        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')

        self.rnn, self.mdn, self.rnn_predictions= self.build_models()

        self.lstm_output, self.hidden_state, self.cell_state= self.rnn(self.inputs)

        self.lstm_output_p, self.hidden_state_p, self.cell_state_p = self.rnn_predictions([self.inputs, self.lstm_inputs_h, self.lstm_inputs_c]) #feed sin predcitions

        self.y_= self.mdn(self.lstm_output)

        self.y_predicted = self.mdn(self.lstm_output_p)
        self.z_loss=self.get_z_loss(self.z_true, self.y_)
        self.reward_loss=self.get_reward_loss(self.z_true, self.y_)


        #self.loss= Z_factor * self.z_loss + R_factor * self.reward_loss #NOT TRAING WITH REWARDS OUTPUTS
        self.loss=self.z_loss
        self.list_gradients = self.rnn.trainable_variables + self.mdn.trainable_variables

        #for op in self.list_gradients:
            #print(op)

        self.Optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5)

        gvs = self.Optimizer.compute_gradients(self.loss)
        #capped_gvs = [(tf.clip_by_value(grad, -CLIP_GRADIENT, CLIP_GRADIENT ), var) for grad, var in gvs]
        capped_gvs=gvs
        self.t_vars=tf.trainable_variables()
        self.train_op = self.Optimizer.apply_gradients(capped_gvs, name='train_step')

        #TODO: add gradient clipping as it as an RNN to avoid back-prop time issues

        self.assign_ops = {}
        for var in self.t_vars:
            print(var)

            # if var.name.startswith('conv_vae'):
            #print(var.name[:-2])
            pshape = var.get_shape()
            pl = tf.placeholder(tf.float32, pshape, var.name[:-2] + '_placeholder')
            print(var.name)
            assign_op = var.assign(pl)
            self.assign_ops[var] = (assign_op, pl)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()
        self.sess=tf.Session()
        self.sess.run(self.init)

        print("Completed creating MDNRNN model")

    def build_models(self):


       lstm_input = tf.keras.layers.Input(shape=(None,self.Z_dim+self.action_dim+1))  #plus 1 is for reward
       lstm_input_h = tf.keras.layers.Input(shape=(None,self.hidden_units))
       lstm_input_c = tf.keras.layers.Input(shape=(self.hidden_units,self.hidden_units))
       print(lstm_input_c)

       lstm = tf.keras.layers.LSTM(self.hidden_units, return_state=True)

       lstm_output, final_hidden_state, final_carry_state = lstm(lstm_input)
       # need to create a copy so that prediction model can input the lstm states during preditions i.e. traning controller
       lstm_output_p, final_hidden_state_p, final_carry_state_p = lstm(lstm_input,
                                                                       initial_state=[lstm_input_h, lstm_input_c])

       mnd_input = tf.keras.layers.Input(shape=(None, self.hidden_units))
       mdn =tf.keras.layers.Dense(self.gaussian_mixtures_number*3*(self.Z_dim))(mnd_input)  #3*as MDN as three output parameter


       lstm_model = tf.keras.Model([lstm_input], [lstm_output, final_hidden_state, final_carry_state])
       predication_model= tf.keras.Model([lstm_input,lstm_input_h, lstm_input_c], [lstm_output, final_hidden_state_p, final_carry_state_p])
       mdn_model = tf.keras.Model([mnd_input], [mdn])

       #print(lstm_model.summary())
       #print(mdn_model.summary)
       print("Print hidden")
       print(final_hidden_state)
       print(lstm_output)

       return lstm_model, mdn_model, predication_model


    def get_z_loss(self, z_true, y_predicted):

        #z_true,reward_true=self.split_outputs(y_true)
        #z_predict=y_predicted[:,:(GMM_DIM*VAE_LATENT_DIM*3)]
        z_predict=y_predicted
        z_predict = tf.reshape(z_predict, [-1, GMM_DIM * 3])

        out_logmix, out_mean, out_logstd = self.get_mdn_coef(z_predict)

        # reshape target data so that it is compatible with prediction shape
        flat_target_data = tf.reshape(z_true, [-1, 1])

        z_loss = self.compute_z_loss(out_logmix, out_mean, out_logstd, flat_target_data)

        return z_loss

    def get_reward_loss(self, y_true, y_pred):
        z_true, rew_true = self.split_outputs(y_true)  # , done_true

        reward_pred = y_pred[:, -1]

        rew_loss = tf.keras.losses.binary_crossentropy(rew_true, reward_pred, from_logits=True)

        rew_loss = tf.reduce_mean(rew_loss)

        return rew_loss

    def tf_lognormal(self, y, mean, logstd):
        return -0.5 * ((y - mean) / tf.exp(logstd)) ** 2 - logstd - logSqrtTwoPI

    def compute_z_loss(self, logmix, mean, logstd, y):

        v = logmix + self.tf_lognormal(y, mean, logstd)
        v = tf.reduce_logsumexp(v, 1, keepdims=True)
        return -tf.reduce_mean(v)



    def get_mdn_coef(self, output):
        logmix, mean, logstd = tf.split(output, 3, 1)
        logmix = logmix - tf.reduce_logsumexp(logmix, 1, keepdims=True)
        return logmix, mean, logstd

    def train(self):

       steps=0;
       #get random batch
       with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as self.sess:

           self.sess.run(self.init)
           for i in range(NUM_STEPS):
               z, z_fs, action, reward = random_batch(filelist, batch_size=BATCH_SIZE)

               steps+=1
               if (steps==NUM_STEPS):
                   break;

               for i in range(BATCH_SIZE):

                   y_in, z_true = get_rnn_inputs(z[i],  action[i], reward[i], z_fs[i])

                   feed_dict = {self.inputs: y_in,
                                self.z_true: z_true,
                                self.learning_rate: LEARNING_RATE}

                   loss, _ = self.sess.run( [self.z_loss,self.train_op], feed_dict=feed_dict)

               print("Step :"+  str(steps)  +  "Loss :" + str(loss))

           self.save_model()

       return


    def predict2(self, rnn_inputs, hidden, cell_state):

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as self.sess:
            a = os.path.join(self.final_model_dir, self.model_name)
            self.saver.restore(self.sess, a)

            feed_dict = {self.inputs: rnn_inputs, self.lstm_inputs_h: hidden, self.lstm_inputs_c: cell_state}

            _, hidden, cell, z_predicted = self.sess.run(
                [self.lstm_output, self.hidden_state, self.cell_state, self.y_predicted],
                feed_dict=feed_dict)

            return hidden, cell, z_predicted

    def predict(self, rnn_inputs, hidden, cell_state):

            feed_dict = {self.inputs: rnn_inputs, self.lstm_inputs_h: hidden, self.lstm_inputs_c: cell_state}

            _, hidden, cell, z_predicted = self.sess.run(
                [self.lstm_output, self.hidden_state, self.cell_state, self.y_predicted],
                feed_dict=feed_dict)

            return hidden, cell, z_predicted

    def save_model(self):

        print ("Saving the model after training")
        if (os.path.exists(self.model_dir)==False):
            #shutil.rmtree(self.model_dir, ignore_errors=True)
            os.makedirs(self.model_dir)

        self.saver.save(self.sess, os.path.join(self.model_dir, self.model_name))
        print("Completed saving the model")


    def load_model(self):
            adir=os.path.join(self.model_dir, self.model_name)
            print ("Checking for the model")
            self.saver.restore(self.sess,adir)
            print ("Session restored")
            self.save_json()


    def step_decay(self, epoch):
        initial_lrate=0.001
        drop = 0.1
        epochs_drop=4
        lrate= initial_lrate* math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate

    def split_outputs(self, y_true):
        z_true = y_true[:, :VAE_LATENT_DIM]
        rew_true = y_true[:, -1]
        # done_true = y_true[:,:,(Z_DIM + 1):]

        return z_true, rew_true  # , done_true

    def get_model_params(self):
        # get trainable params.
        model_names = []
        model_params = []
        model_shapes = []


        for var in self.t_vars:
                # if var.name.startswith('conv_vae'):
                param_name = var.name
                p = self.sess.run(var)
                model_names.append(param_name)
                params = np.round(p * 10000).astype(np.int).tolist()
                model_params.append(params)
                model_shapes.append(p.shape)
        return model_params, model_shapes, model_names


    def set_model_params(self, params):
        #with self.g.as_default():
            #t_vars = tf.trainable_variables()
            idx = 0
            for var in self.t_vars:
                print("Restoring RNN" )
                print(var)
                # if var.name.startswith('conv_vae'):
                pshape = tuple(var.get_shape().as_list())
                p = np.array(params[idx])
                print(pshape)
                print(p.shape)
                assert pshape == p.shape, "inconsistent shape"
                assign_op, pl = self.assign_ops[var]
                self.sess.run(assign_op, feed_dict={pl.name: p / 10000.})
                idx += 1

    def load_json(self, jsonfile='rnn.json'):
        with open(jsonfile, 'r') as f:
            params = json.load(f)

        self.set_model_params(params)

    def save_json(self, jsonfile='rnn2.json'):
        model_params, model_shapes, model_names = self.get_model_params()
        qparams = []
        for p in model_params:
            qparams.append(p)
        with open(jsonfile, 'wt') as outfile:
            json.dump(qparams, outfile, sort_keys=True, indent=0, separators=(',', ': '))



if __name__ == '__main__':

    model=MDNRNN()
    #model.save_json()
    #model.load_json()
    model.load_model()


