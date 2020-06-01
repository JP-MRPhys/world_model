import tensorflow as tf
import numpy as np
import shutil
import os
import math
# useful referene for MDN

VAE_LATENT_DIM=64
ACTION_DIM=2
RNN_HIDDEN_UNITS=256
GMM_DIM=5
logSqrtTwoPI = np.log(np.sqrt(2.0 * np.pi))

class MDNRNN():

    def __init__(self):


        self.Z_dim= VAE_LATENT_DIM
        self.action_dim = ACTION_DIM
        self.gaussian_mixtures_number= GMM_DIM
        self.hidden_units= RNN_HIDDEN_UNITS

        self.model_name='MDNRNN'
        self.logdir = './trained_models/' + self.model_name  # if not exist create logdir
        self.model_dir = self.logdir + '/final_model'


        self.inputs = tf.placeholder(tf.float32, shape=[None, None, self.Z_dim+self.action_dim])
        self.y = tf.placeholder(tf.float32, shape=[None, None, self.Z_dim])
        self.learning_rate = tf.placeholder(tf.float32, [], name='learning_rate')


        self.rnn, self.mdn=self.build_models()

        self.lstm_output, self.hidden_state, self.cell_state= self.rnn(self.inputs)
        #self.lstm_output_reshape = tf.reshape(self.lstm_output, [-1, self.hidden_units])
        self.mdn_output=self.mdn(self.lstm_output)
        self.mdn_output_reshape = tf.reshape(self.mdn_output, [-1, self.gaussian_mixtures_number*3])

        out_logmix, out_mean, out_logstd = self.get_mdn_coef(self.mdn_output_reshape)

        # reshape target data so that it is compatible with prediction shape
        flat_target_data = tf.reshape(self.y, [-1, 1])

        self.loss = self.get_loss(out_logmix, out_mean, out_logstd, flat_target_data)

        self.cost = tf.reduce_mean(self.loss)



        self.list_gradients = self.rnn.trainable_variables + self.mdn.trainable_variables

        #for op in self.list_gradients:
            #print(op)
        self.Optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate, beta1=0.5).minimize(self.loss, var_list=self.list_gradients)

        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

        print("Completed createing the model")

    def build_models(self):



       lstm_input = tf.keras.layers.Input(shape=(None,self.Z_dim+self.action_dim+1))  #plus 1 is for reward
       lstm_ouput, final_hidden_state, final_carry_state =tf.keras.layers.LSTM(self.hidden_units, return_state=True)(lstm_input)

       mnd_input = tf.keras.layers.Input(shape=(None, self.hidden_units))
       mdn =tf.keras.layers.Dense(self.gaussian_mixtures_number*3*(self.Z_dim))(mnd_input)  #3*as MDN as three output parameter


       lstm_model = tf.keras.Model([lstm_input], [lstm_ouput, final_hidden_state, final_carry_state])
       mdn_model = tf.keras.Model( [mnd_input], [mdn])

       print(lstm_model.summary())
       print(mdn_model.summary)
       print("Print hidden")
       print(final_hidden_state)


       return lstm_model, mdn_model



    def tf_lognormal(self, y, mean, logstd):
        return -0.5 * ((y - mean) / tf.exp(logstd)) ** 2 - logstd - logSqrtTwoPI

    def get_loss(self, logmix, mean, logstd, y):
        v = logmix + self.tf_lognormal(y, mean, logstd)
        v = tf.reduce_logsumexp(v, 1, keepdims=True)
        return -tf.reduce_mean(v)

    def get_mdn_coef(self, output):
        logmix, mean, logstd = tf.split(output, 3, 1)
        logmix = logmix - tf.reduce_logsumexp(logmix, 1, keepdims=True)
        return logmix, mean, logstd

    def train(self):
        return


    def predict(self):
        return


    def save_model(self, model_name):

        print ("Saving the model after training")
        if (os.path.exists(self.model_dir)):
            shutil.rmtree(self.model_dir, ignore_errors=True)
            os.makedirs(self.model_dir)


        self.saver.save(self.sess, os.path.join(self.model_dir, self.model_name))
        print("Completed saving the model")



    def load_model(self, model_name):

        print ("Checking for the model")

        with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as new_sess:
            saver =tf.train.import_meta_graph((model_name + '.meta'))
            #saver.restore(self.sess, self.model_dir)
            saver.restore(new_sess,tf.train.latest_checkpoint("./"))
            print ("Session restored")
            return new_sess

    def step_decay(self, epoch):
        initial_lrate=0.001
        drop = 0.5
        epochs_drop=4
        lrate= initial_lrate* math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lrate


if __name__ == '__main__':

    model=MDNRNN()