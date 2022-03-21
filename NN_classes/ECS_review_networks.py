'''
Created on 21 03 2022

@author: ManuLahariya
'''
import tensorflow as tf
from tensorflow.python.ops import rnn_cell, rnn
import numpy as np
from NN_classes.NN_class import NN_class
from NN_classes.TF_ECS_tower_model import Calculate_NonLinearOP
import time

np.random.seed(1234)
tf.set_random_seed(1234)


class LSTM(NN_class):
    '''
    Physics based Long Short-Term Memory Network for RC circuit.
    Predict, Network and Loss methods are overwritten.
    'teacher forcing' is used for training
    '''
    def __init__(self, t, X, Y, layers,
                 train_on_last=True,
                 activation = tf.keras.activations.sigmoid):
        '''
        time_steps is the length of sequence that needs to be passed through lstm to get
        prediction for the next time slot.
        '''
        NN_class.__init__(self, t, X, Y, layers,
                          train_on_last,
                          activation)
        self.Model_name = 'PyLSTM'
        self.time_steps = int(4 * 15) # 1 hours
        self.batchsize = 30 * 24 * 4 * 15 # one month per batch
        self.prepare_session()


    def initialize_NN(self, layers):
        weights = []
        biases = []
        W = self.xavier_init(size=[layers[-2], layers[-1]])
        b = tf.Variable(tf.zeros([1, layers[-1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)
        return weights, biases

    def Loss(self, X_tf, t_tf, Y_tf):
        '''
        :param Y_pred: Predicted values of y
        :param Y_tf: real values of y
        :param f_y_pred: f function values
        :return: loss: MSEt = MSEy + MSEf
        Note: network returns Y_total-timesteps fitted values. This happens because the fist timestep values are
        used in the network
        '''
        Y_pred, f_Y_pred = self.Network(X=self.X_tf, Y_prev=self.Y_tf[:-1, :], t=self.t_tf)
        self.Y_pred, self.f_Y_pred = Y_pred, f_Y_pred
        if self.train_on_last: Y_tf_orignal = Y_tf[self.time_steps:]
        else: Y_tf_orignal = Y_tf
        Norm_Y = (Y_tf_orignal - self.Ymean) / self.Ysd

        losses = tf.square(Norm_Y - Y_pred)

        final_loss = tf.reduce_mean(losses + 0.005 * tf.reduce_sum(
            [tf.nn.l2_loss(n) for n in tf.trainable_variables() if 'bias' not in n.name]))
        return final_loss


    def make_cell(self,lstm_size):
        return tf.nn.rnn_cell.BasicLSTMCell(lstm_size,activation=self.activation , state_is_tuple=True)

    def LSTM(self, X,t, weights, biases):
        # num_layers = len(weights) + 1
        # X.shape = batch_size, chunk,size
        X = (X - self.Xmean) / self.Xsd
        H = tf.concat((X,t),axis=1)

        if self.train_on_last:
            H = tf.cond(self.Train_tf,
                          lambda: tf.concat([H[i:-(self.time_steps-i), :] for i in range(0, self.time_steps)], axis=0)
                          ,lambda: H)
            H = tf.split(value=H, num_or_size_splits=self.time_steps, axis=0)

            network = rnn_cell.MultiRNNCell([self.make_cell(self.layers[i + 1]) for i in range(len(self.layers) - 2)],
                                            state_is_tuple=True)
            outputs, state = rnn.static_rnn(network, H, dtype=tf.float32)
            outputs = outputs[-1]
        else:
            H = tf.reshape(H, [-1, self.time_steps, self.layers[0]])
            H = tf.transpose(H, [1, 0, 2])
            H = tf.reshape(H, [-1, self.layers[0]])
            H = tf.split(value=H,num_or_size_splits=self.time_steps,axis=0)
            network = rnn_cell.MultiRNNCell([self.make_cell(self.layers[i+1]) for i in range(len(self.layers)-2)],
                                            state_is_tuple=True)
            outputs, state = rnn.static_rnn(network,H,dtype=tf.float32)
            outputs = tf.concat(outputs,axis=0)
            outputs = tf.reshape(outputs, [self.time_steps, -1,outputs.shape[1]])
            outputs = tf.transpose(outputs,[1,0,2])
            outputs = tf.reshape(outputs,[-1,outputs.shape[2]])

        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(outputs, W), b)
        return Y

    def Network(self, X,t,Y_prev=None):
        '''
           Modified network:
            :returns voltage across capacitor:y
            :returns :f = dy/dt +
        '''
        first_val = tf.reshape(Y_prev[0,:],[1,1])
        y_out = self.LSTM(X,t, self.weights, self.biases)
        y_unnorm = (y_out * self.Ysd + self.Ymean)[:, 0:1]
        # use each output to calculate loss
        y_t =  tf.gradients(y_unnorm,t)[0]
        wd = self.Weather_data_tf

        if self.train_on_last:
            y_t = y_t[self.time_steps:]
            X = X[self.time_steps:]
            wd = wd[self.time_steps:]

        self.NonlinOP = Calculate_NonLinearOP(Fan_powers = X,
                                              Weather_data_tf = wd,
                                              T_basin = y_unnorm)
        # calculate F
        f = y_t - self.NonlinOP

        return y_out, f

    def predict(self,  X_star, t_star, wd_star=None, Iniital=0):
        '''
        output is intilized with y_initial for initial time_steps, after which
        prediction is made for each time slot
        :return: y: predctions
        We predict for all values in X_star, t_star. For the first prediction, we use the initial values of
        X_Star and t_star for time_slot times.
        Each prediction is used as an input for the next time slot.
        '''
        if wd_star is None: wd_star = np.zeros((X_star.shape[0], 4))
        if isinstance( Iniital, list):
            y_star = Iniital[0].reshape(-1, 1)
            X_star = np.concatenate((Iniital[1], X_star))
            t_star = np.concatenate((Iniital[2], t_star))
            wd_star = np.concatenate(
                (self.train_weather_data[-self.time_steps:].reshape(-1, self.train_weather_data.shape[1]),
                 wd_star))
        else:
            y_star = np.repeat(float(Iniital),self.time_steps).reshape(-1,1)
            X_star = np.concatenate( (np.tile(X_star[0,:], self.time_steps).reshape(-1, X_star.shape[1]), X_star))
            t_star = np.concatenate( (np.linspace(t_star[0,:]-self.time_steps,t_star[0,:], self.time_steps).reshape(-1, t_star.shape[1]), t_star))
            wd_star = np.concatenate(
                (np.tile(wd_star[0, :], self.time_steps).reshape(-1, wd_star.shape[1]),
                 wd_star))

        pred_time = []
        t = []
        y_pred = []
        for i in range(X_star.shape[0] - self.time_steps):
            start = time.time()
            idx = np.arange(i,i + self.time_steps)
            tf_dict = self.prep_tf_dict(X=X_star[idx, :], t=t_star[idx, :],
                                        Y=y_star[idx, :], wd = wd_star[idx, :],
                                        Train=False)
            y = self.sess.run(self.Y_pred, tf_dict)
            if not(self.train_on_last): y = y[-1]
            y = y * self.Ysd + self.Ymean
            end = time.time()
            y_star = np.append(y_star,y).reshape(-1,1)
            t.append(float(t_star[i+self.time_steps, :]))
            y_pred.append(float(y.flatten()))
            pred_time.append(end - start)

        return np.column_stack([t,y_pred,pred_time])



class PyLSTMwof(NN_class):
    '''
    Physics based Long Short-Term Memory Network for RC circuit.
    Predict, Network and Loss methods are overwritten.
    'teacher forcing' is used for training
    '''
    def __init__(self, t, X, Y, layers,
                 train_on_last=True,
                 activation = tf.keras.activations.sigmoid):
        '''
        time_steps is the length of sequence that needs to be passed through lstm to get
        prediction for the next time slot.
        '''
        NN_class.__init__(self, t, X, Y, layers,
                          train_on_last,
                          activation)
        self.Model_name = 'PyLSTM'
        self.time_steps = int(4 * 15) # 1 hours
        self.batchsize = 30 * 24 * 4 * 15 # one month per batch
        self.prepare_session()


    def initialize_NN(self, layers):
        weights = []
        biases = []
        W = self.xavier_init(size=[layers[-2], layers[-1]])
        b = tf.Variable(tf.zeros([1, layers[-1]], dtype=tf.float32), dtype=tf.float32)
        weights.append(W)
        biases.append(b)
        return weights, biases

    def Loss(self, X_tf, t_tf, Y_tf):
        '''
        :param Y_pred: Predicted values of y
        :param Y_tf: real values of y
        :param f_y_pred: f function values
        :return: loss: MSEt = MSEy + MSEf
        Note: network returns Y_total-timesteps fitted values. This happens because the fist timestep values are
        used in the network
        '''
        Y_pred, f_Y_pred = self.Network(X=self.X_tf, Y_prev=self.Y_tf[:-1, :], t=self.t_tf)
        self.Y_pred, self.f_Y_pred = Y_pred, f_Y_pred
        if self.train_on_last: Y_tf_orignal = Y_tf[self.time_steps:]
        else: Y_tf_orignal = Y_tf
        Norm_Y = (Y_tf_orignal - self.Ymean) / self.Ysd
        losses = tf.square(Norm_Y - Y_pred) + tf.square(f_Y_pred)
        final_loss = tf.reduce_mean(losses + 0.005 * tf.reduce_sum(
            [tf.nn.l2_loss(n) for n in tf.trainable_variables() if 'bias' not in n.name]))
        return final_loss


    def make_cell(self,lstm_size):
        return tf.nn.rnn_cell.BasicLSTMCell(lstm_size,activation=self.activation , state_is_tuple=True)

    def LSTM(self, X,t, weights, biases):
        # num_layers = len(weights) + 1
        # X.shape = batch_size, chunk,size
        X = (X - self.Xmean) / self.Xsd
        H = tf.concat((X,t),axis=1)

        if self.train_on_last:
            H = tf.cond(self.Train_tf,
                          lambda: tf.concat([H[i:-(self.time_steps-i), :] for i in range(0, self.time_steps)], axis=0)
                          ,lambda: H)
            H = tf.split(value=H, num_or_size_splits=self.time_steps, axis=0)

            network = rnn_cell.MultiRNNCell([self.make_cell(self.layers[i + 1]) for i in range(len(self.layers) - 2)],
                                            state_is_tuple=True)
            outputs, state = rnn.static_rnn(network, H, dtype=tf.float32)
            outputs = outputs[-1]
        else:
            H = tf.reshape(H, [-1, self.time_steps, self.layers[0]])
            H = tf.transpose(H, [1, 0, 2])
            H = tf.reshape(H, [-1, self.layers[0]])
            H = tf.split(value=H,num_or_size_splits=self.time_steps,axis=0)
            network = rnn_cell.MultiRNNCell([self.make_cell(self.layers[i+1]) for i in range(len(self.layers)-2)],
                                            state_is_tuple=True)
            outputs, state = rnn.static_rnn(network,H,dtype=tf.float32)
            outputs = tf.concat(outputs,axis=0)
            outputs = tf.reshape(outputs, [self.time_steps, -1,outputs.shape[1]])
            outputs = tf.transpose(outputs,[1,0,2])
            outputs = tf.reshape(outputs,[-1,outputs.shape[2]])

        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(outputs, W), b)
        return Y

    def Network(self, X,t,Y_prev=None):
        '''
           Modified network:
            :returns voltage across capacitor:y
            :returns :f = dy/dt +
        '''
        first_val = tf.reshape(Y_prev[0,:],[1,1])
        y_out = self.LSTM(X,t, self.weights, self.biases)
        y_unnorm = (y_out * self.Ysd + self.Ymean)[:, 0:1]
        # use each output to calculate loss
        y_t =  tf.gradients(y_unnorm,t)[0]
        wd = self.Weather_data_tf

        if self.train_on_last:
            y_t = y_t[self.time_steps:]
            X = X[self.time_steps:]
            wd = wd[self.time_steps:]

        self.NonlinOP = Calculate_NonLinearOP(Fan_powers = X,
                                              Weather_data_tf = wd,
                                              T_basin = y_unnorm)
        # calculate F
        f = y_t - self.NonlinOP

        return y_out, f

    def predict(self,  X_star, t_star, wd_star=None, Iniital=0):
        '''
        output is intilized with y_initial for initial time_steps, after which
        prediction is made for each time slot
        :return: y: predctions
        We predict for all values in X_star, t_star. For the first prediction, we use the initial values of
        X_Star and t_star for time_slot times.
        Each prediction is used as an input for the next time slot.
        '''
        if wd_star is None: wd_star = np.zeros((X_star.shape[0], 4))
        if isinstance( Iniital, list):
            y_star = Iniital[0].reshape(-1, 1)
            X_star = np.concatenate((Iniital[1], X_star))
            t_star = np.concatenate((Iniital[2], t_star))
            wd_star = np.concatenate(
                (self.train_weather_data[-self.time_steps:].reshape(-1, self.train_weather_data.shape[1]),
                 wd_star))
        else:
            y_star = np.repeat(float(Iniital),self.time_steps).reshape(-1,1)
            X_star = np.concatenate( (np.tile(X_star[0,:], self.time_steps).reshape(-1, X_star.shape[1]), X_star))
            t_star = np.concatenate( (np.linspace(t_star[0,:]-self.time_steps,t_star[0,:], self.time_steps).reshape(-1, t_star.shape[1]), t_star))
            wd_star = np.concatenate(
                (np.tile(wd_star[0, :], self.time_steps).reshape(-1, wd_star.shape[1]),
                 wd_star))

        pred_time = []
        t = []
        y_pred = []
        for i in range(X_star.shape[0] - self.time_steps):
            start = time.time()
            idx = np.arange(i,i + self.time_steps)
            tf_dict = self.prep_tf_dict(X=X_star[idx, :], t=t_star[idx, :],
                                        Y=y_star[idx, :], wd = wd_star[idx, :],
                                        Train=False)
            y = self.sess.run(self.Y_pred, tf_dict)
            if not(self.train_on_last): y = y[-1]
            y = y * self.Ysd + self.Ymean
            end = time.time()
            y_star = np.append(y_star,y).reshape(-1,1)
            t.append(float(t_star[i+self.time_steps, :]))
            y_pred.append(float(y.flatten()))
            pred_time.append(end - start)

        return np.column_stack([t,y_pred,pred_time])





