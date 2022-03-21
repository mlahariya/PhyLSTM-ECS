'''
Created on 21 03 2022

@author: ManuLahariya
'''

import tensorflow as tf
import numpy as np
import time

np.random.seed(1234)
tf.set_random_seed(1234)

class NN_class:
    '''
       Deep Network class for system identification
       This is base class that is inherited in NN, PyNN and PyLSTM
       Some methods are overwritten in PyLSTM and PyNN
    '''
    # Initialize the class
    def __init__(self, t, X, Y, layers,
                 train_on_last,
                 activation):
        '''
            :param t : np array of time of observation
            :param X : np array of inputs to the network
            :param Y : np array of outputs of the network
            :param layers : list depicting the layers where
                length(layers): length of layers
                layers[i]: number of nodes in ith layer
        '''
        self.Y = Y
        self.X = X
        self.t = t
        self.layers = layers
        self.Xmean = self.X.mean(0)
        self.Xsd = self.X.std(0)
        self.Ymean = self.Y.mean(0)
        self.Ysd = self.Y.std(0)
        self.layers = layers
        self.batchsize = 100 * 100
        self.train_weather_data = np.zeros((X.shape[0],4))
        self.train_on_last = train_on_last
        self.use_scipy_opt = True
        self.activation = activation

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def prepare_session(self):
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(self.layers)
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.X_tf = tf.placeholder(tf.float32, shape=[None, self.X.shape[1]])
        self.t_tf = tf.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        self.Y_tf = tf.placeholder(tf.float32, shape=[None, self.Y.shape[1]])
        self.Weather_data_tf = tf.placeholder(tf.float32, shape=[None, 4])
        self.Train_tf = tf.placeholder(tf.bool)  # this inficates if we are training
        self.loss = self.Loss(X_tf = self.X_tf, Y_tf= self.Y_tf, t_tf = self.t_tf)
        # we use two optimizers! adam first than scipy
        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method = 'L-BFGS-B',
                                                                options = {'maxiter': 2000,
                                                                           'maxfun': 2000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})
        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def save_session(self, loc):
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, loc)

    def load_session(self, loc):
        saver = tf.train.Saver()
        saver.restore(self.sess, loc)

    def callback(self, loss):
        print(self.Model_name+': It: %d, Loss: %.4e (scipy external opt)' % (self.iter_for_scipy, loss))
        self.iter_for_scipy = 1 + self.iter_for_scipy


    def batch(self, iterable, size=1):
        l = len(iterable)
        for ndx in range(0, l, size):
            yield iterable[ndx:min(ndx + size, l)]

    def train(self, nIter):
        '''
            :param nIter: number of iterations to train the network for
        '''
        iteration_times = []
        iteration = []
        batch = []
        loss = []
        for it in range(nIter):
            b = 1
            for idx in self.batch(range(self.X.shape[0]), int(self.batchsize)):
                self.idx = idx
                iter_start_time = time.time()
                tf_dict = self.prep_tf_dict(self.X[idx, :], self.t[idx, :], self.Y[idx, :],
                                            wd=self.train_weather_data[idx,:])
                self.sess.run(self.train_op_Adam, tf_dict)
                iter_end_time = time.time()
                # Print
                elapsed = iter_end_time - iter_start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print(self.Model_name + ': It: %d, Loss: %.4e, Time: %.2f , batch idx: %d:%d (tf adam opt)' %
                      (it, loss_value, elapsed, min(idx), max(idx)))
                iteration_times.append(elapsed)
                iteration.append(it)
                batch.append(b)
                loss.append(loss_value)
                b = b+1

        if self.use_scipy_opt:
            self.iter_for_scipy = 1
            tf_dict = self.prep_tf_dict(self.X, self.t, self.Y,wd=self.train_weather_data)
            self.optimizer.minimize(self.sess,
                                    feed_dict=tf_dict,
                                    fetches=[self.loss],
                                    loss_callback=self.callback)

        return np.column_stack([iteration,batch,iteration_times,loss])

    def load_weather_data(self,train_data):
        '''
        :param file_name: loc and file name for the dataset
        :return: saves the weather data for the simulation in our dataset
        Note: We use weather data to calculate mass flow rate of air, we assume we have perfect knowledge of weather
        '''
        self.train_weather_data = train_data

    def prep_tf_dict(self, X, t, Y=None,wd=None, Train=True):
        tf_dict = {self.X_tf: X, self.t_tf: t, self.Train_tf:Train,
                   self.Weather_data_tf: wd}
        if Y is not None: tf_dict.update({self.Y_tf: Y})
        return tf_dict

    '''
        These methods can be overwritten in derived classes
    '''
    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def Loss(self, X_tf, t_tf, Y_tf):
        Y_pred, f_Y_pred = self.Network(X=X_tf, t=t_tf)
        self.Y_pred, self.f_Y_pred = Y_pred, f_Y_pred
        Norm_Y = (Y_tf - self.Ymean)/self.Ysd
        losses = tf.square(Norm_Y - Y_pred)
        final_loss = tf.reduce_mean(losses + 0.005 * tf.reduce_sum(
            [tf.nn.l2_loss(n) for n in tf.trainable_variables() if 'bias' not in n.name]))
        return final_loss

    def neural_net(self, X, t , weights, biases):
        num_layers = len(weights) + 1
        H = (X - self.Xmean) / self.Xsd
        H = tf.concat([H,t],1)
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = self.activation(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def Network(self, X,t,Y_prev=None):
        y_out = self.neural_net(X, t, self.weights, self.biases)
        y = y_out[:, 0:1]
        return y, y

    def predict(self, X_star, t_star,wd_star=None, Iniital=0):
        '''
            :param X_star: X for predictions
            :param t_star: t for predictions
            :param y_initial: initial value of y (used for PyLSTM)
            This method is overwritten in PyLSTM
        '''
        t = []
        y_pred = []
        pred_time = []
        if wd_star is None: wd_star = np.zeros((X_star.shape[0],4))
        for i in range(X_star.shape[0]):
            start = time.time()
            tf_dict = self.prep_tf_dict(X =X_star[i,:].reshape(1,-1),
                                        t= t_star[i,:].reshape(1,-1),
                                        wd = wd_star[i,:].reshape(1,-1),
                                        Train=False)
            Y_star = self.sess.run(self.Y_pred, tf_dict)
            Y_star = Y_star*self.Ysd + self.Ymean
            end = time.time()
            t.append(float(t_star[i,:]))
            y_pred.append(float(Y_star))
            pred_time.append(end-start)
        return np.column_stack([t,y_pred,pred_time])



