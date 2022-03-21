# -------------------------------------- Manu Lahariya 18/08/2020 ------------------------------------
#
# Cross validation of all models for all months
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
from NN_classes.ECS_review_networks import PyLSTMwof, LSTM
import csv
import os

slot_per_hour = 60
hour_to_month = 24*30
Test_months = 1
N_validations =5
models = [NN, PyNN, PyLSTM, LSTM, PyLSTMwof]
mnames = ['NN','PyNN', 'PyLSTM', 'LSTM', "PyLSTMwof"]
iterations = 1
Train_months_arr = [1,3,5,7]
locations = ['res/predictions/ECS/1 month',
             'res/predictions/ECS/3 month',
             'res/predictions/ECS/5 month',
             'res/predictions/ECS/7 month']




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

for p in range(0,len(Train_months_arr)):
    Train_months = Train_months_arr[p]
    save_loc = locations[p]
    if not os.path.exists(save_loc): os.makedirs(save_loc)

    # Moving Window
    N_train_slots = slot_per_hour * hour_to_month * Train_months
    Increment = slot_per_hour * hour_to_month * Test_months
    N_test_slots = slot_per_hour * hour_to_month
    start_from = slot_per_hour * 24

    # timelogs
    train_file_path = os.path.join(save_loc, 'PyBasedNN_training_results.csv')
    with open(train_file_path, 'w') as csvfile:
        writ = csv.writer(csvfile, delimiter=',', lineterminator='\n', )
        writ.writerow(['Validation_set','Train_months', 'Test_months','Model','Layers','Iter','Batch','Time','Loss'])

    fitted_file_path = os.path.join(save_loc, 'PyBasedNN_fitting_results.csv')
    with open(fitted_file_path, 'w') as csvfile:
        writ = csv.writer(csvfile, delimiter=',', lineterminator='\n', )
        writ.writerow(['Validation_set','Train_months', 'Test_months','Model','Layers','Pred_slot','Pred_dt','Y','Y_noise','Y_pred','Pred_time'])

    test_pred_file_path = os.path.join(save_loc, 'PyBasedNN_test_predictions_results.csv')
    with open(test_pred_file_path, 'w') as csvfile:
        writ = csv.writer(csvfile, delimiter=',', lineterminator='\n', )
        writ.writerow(['Validation_set','Train_months', 'Test_months','Model','Layers','Pred_slot','Pred_dt','Y','Y_noise','Y_pred','Pred_time'])


    data = data.iloc[start_from:]
    X_cols = ['Power_fan_1', 'Power_fan_2']
    T_cols = ['hour_of_day']  # this is the time slot of the recorded data
    Y_noisy_cols = ['Tb_noise']
    Y_cols = ['Tb']
    for i in np.arange(0,N_validations):
        Train_data = data.iloc[i*Increment:(i*Increment)+N_train_slots,:].sort_values('time')
        Test_data = data.iloc[(i*Increment)+N_train_slots:(i*Increment)+(N_train_slots+N_test_slots),:].sort_values('time')

        train_weather_data = np.array( weather_data.loc[weather_data['time'].isin(Train_data['time'])].sort_values('time') )
        test_weather_data = np.array( weather_data.loc[weather_data['time'].isin(Test_data['time'])].sort_values('time') )

        Train_dt, Test_dt = np.array(Train_data[['Date_time']]), np.array(Test_data[['Date_time']])
        t_train, t_test = np.array(Train_data[T_cols]), np.array(Test_data[T_cols])
        X_train, X_test = np.array(Train_data[X_cols]), np.array(Test_data[X_cols])
        Y_train, Y_test = np.array(Train_data[Y_cols]), np.array(Test_data[Y_cols])
        Y_noise_train, Y_noise_test = np.array(Train_data[Y_noisy_cols]), np.array(Test_data[Y_noisy_cols])

        for model_indicator in range(2,3):
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

            # saving logs
            with open(train_file_path, 'a') as csvfile:
                writ = csv.writer(csvfile, delimiter=',', lineterminator='\n', )
                for j in range(Training_returns.shape[0]):
                    writ.writerow([i+1,Train_months,Test_months,
                                   mname, layers,
                                   Training_returns[j,0],Training_returns[j,1],
                                   Training_returns[j,2],Training_returns[j,3]])


            # --------------------------------------------------------------------------------------------------
            # Predictions
            # --------------------------------------------------------------------------------------------------
            # testing period
            initial = [Y_train[-slot_per_hour:], X_train[-slot_per_hour:], t_train[-slot_per_hour:]]
            pred_returns = model.predict(X_test, t_test, Iniital=initial)
            # saving logs
            with open(test_pred_file_path, 'a') as csvfile:
                writ = csv.writer(csvfile, delimiter=',', lineterminator='\n', )
                for j in range(pred_returns.shape[0]):
                    writ.writerow([i + 1, Train_months, Test_months,
                                   mname, layers,
                                   pred_returns[j, 0], Test_dt[j, 0],
                                   Y_test[j, 0], Y_noise_test[j, 0],
                                   pred_returns[j, 1], pred_returns[j, 2]])

            # fitting
            pred_returns = model.predict(X_train, t_train, Iniital = Y_train[0])

            # saving logs
            with open(fitted_file_path, 'a') as csvfile:
                writ = csv.writer(csvfile, delimiter=',', lineterminator='\n', )
                for j in range(pred_returns.shape[0]):
                    writ.writerow([i+1,Train_months,Test_months,
                                   mname, layers,
                                   pred_returns[j,0],Train_dt[j,0],
                                   Y_train[j,0],Y_noise_train[j,0],
                                   pred_returns[j,1],pred_returns[j,2]])

