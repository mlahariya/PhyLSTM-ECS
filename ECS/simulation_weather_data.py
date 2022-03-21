'''
Created on 21 03 2022

@author: ManuLahariya
'''

import pandas as pd
import os
save_loc = '../res/simulation'

df_weather = pd.read_pickle(os.path.join(save_loc, 'weather2017.pkl'))
df_simulated = pd.read_csv('../res/simulation/data/ECS_Tower_simulated_data-35040-2021-01-24-19-47.csv')

df_sim = df_simulated[['Weather_dt','time']]
df_sim.index = df_sim['Weather_dt']
df_wet = pd.DataFrame({'Weather_dt':df_weather.index,
                       'temp':df_weather['temp'],
                       'hum':df_weather['hum'],
                       'pres':df_weather['pres']})

df_full = pd.merge(df_sim, df_wet, left_index=True, right_index=True, how='inner')
df_to_save = df_full[['time','temp','hum','pres']]
df_to_save.to_csv(os.path.join(save_loc,'data','Weatherdata_forfile-2021-01-24-19-47.csv'),index=False)