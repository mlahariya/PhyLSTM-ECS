# -------------------------------------- Manu Lahariya 18/08/2020 ------------------------------------
#
# Training and saving a model
#
# ------------------------------------------------------------------------------------------------------
'''
Created on 21 03 2022

@author: ManuLahariya
'''

import pandas as pd
import numpy as np
import tensorflow as tf
from NN_classes.ECS_networks import NN, PyNN, PyLSTM
import os
import datetime

slot_per_hour = 60
hour_to_month = 24*30
models = [NN, PyNN, PyLSTM]
mnames = ['NN','PyNN', 'PyLSTM']
iterations = 1
locations = ['res/models/ECS/NN',
             'res/models/ECS/PyNN',
             'res/models/ECS/PyLSTM']

# --------------------------------------------------------------------------------------------------
# Reading and preparing data
# --------------------------------------------------------------------------------------------------
data = pd.read_csv('res/simulation/data/ECS_Tower_simulated_data-35040-2020-11-28-00-02.csv')
weather_data = pd.read_csv('res/simulation/data/Weatherdata_forfile_2020-11-28-00-02.csv')
data[['hour_of_day']] = pd.DataFrame([(x.hour + x.minute/60) for x in pd.to_datetime(data['Date_time'])])
save_d = []
train_sample = []
test_sample = []
activation = tf.keras.activations.sigmoid

# Data set
start_from = slot_per_hour * 24
data = data.iloc[start_from:]
X_cols = ['Power_fan_1', 'Power_fan_2']
T_cols = ['hour_of_day']  # this is the time slot of the recorded data
Y_noisy_cols = ['Tb_noise']
Y_cols = ['Tb']

Train_data = data.iloc[:,:].sort_values('time')
train_weather_data = np.array(weather_data.loc[weather_data['time'].isin(Train_data['time'])].sort_values('time') )

Train_dt = np.array(Train_data[['Date_time']])
t_train = np.array(Train_data[T_cols])
X_train = np.array(Train_data[X_cols])
Y_train = np.array(Train_data[Y_cols])
Y_noise_train = np.array(Train_data[Y_noisy_cols])

for model_indicator in range(0,3):
    mname = mnames[model_indicator]
    layers = [4, 16, 16, 1]
    if mname != 'PyLSTM': layers = [3, 16, 16, 1]

    # --------------------------------------------------------------------------------------------------
    # Training and saving logs
    # --------------------------------------------------------------------------------------------------

    # create models
    tf.reset_default_graph()
    model_class = models[model_indicator]

    model = model_class(t=t_train,
                        X=X_train,
                        Y=Y_noise_train,
                        layers=layers,
                        train_on_last=False,
                        activation=activation)

    if mname != 'NN': model.load_weather_data(train_data = train_weather_data)
    model.use_scipy_opt = False

    # Training
    Training_returns = model.train(iterations)

    # Saving
    save_loc = locations[0]
    if not os.path.exists(save_loc): os.makedirs(save_loc)
    save_name = datetime.datetime.now().strftime('model-'+mname+'-'+str(iterations)+ '-%Y-%m-%d-%H-%M')
    loc_name = os.path.join(save_loc, save_name)
    model.save_session(loc=loc_name)



