# -------------------------------------- Manu Lahariya 18/08/2020 ------------------------------------
#
# Simulate data for ECS circuit using simpy
#
# ------------------------------------------------------------------------------------------------------
'''
Created on 21 03 2022

@author: ManuLahariya
'''

import simpy
import numpy as np
import random
import pandas as pd
import os
import datetime
from plots.simulation_plots import plot_simulated_date
from ECS.ECS_tower_model import T_basin_calculation

## Parameters
total_time = 365*24*4 # 15 minute slots and data for 1 year
# Change the parameters with in this range of seconds
Change_slots = np.arange(4*24,4*4*24,4*6)
# values of V,R and C
# these values are randomly changed during the simulation
P_fan_1 = np.array((110,120,130,140,150,160,170,180,190,200))
P_fan_2 = np.array((110,120,130,140,150,160,170,180,190,200))
Q_process = np.array((33,33)) #np.array((22,29.33,36.66,44))
save_loc = '../res/simulation'
save_name = datetime.datetime.now().strftime('ECS_Tower_simulated_data-'+str(total_time)+'-%Y-%m-%d-%H-%M')
plot_sec = 500
data = []

def main():
    env = simpy.Environment()
    env.process(RC_control(env))
    # time is unit less in simpy. we use it as seconds
    env.run(total_time)
    print(data)
    save_data = pd.concat(data)
    time = np.arange(0, save_data.shape[0])
    save_data['time'] = time
    save_data = save_data[['Date_time','time','Tb','Tb_noise','Power_fan_1','Power_fan_2','Q_process','Weather_dt']]
    plt = plot_simulated_date(data = save_data,title=save_name,plt_x_secs = plot_sec)
    plt.savefig(os.path.join(save_loc,'plots', save_name+'.png'))
    save_data.to_csv(os.path.join(save_loc,'data', save_name+'.csv'),index=False)
    print("Simulation Done")


def RC_control(env):
    Tb_initial = 0
    df_weather = pd.read_pickle(os.path.join(save_loc, 'weather2017.pkl'))
    weather_slot = 0
    while True:
        p1 = random.choice(P_fan_1)
        p2 = random.choice(P_fan_2)
        q = random.choice(Q_process)
        t= random.choice(Change_slots)
        if weather_slot + t > 35039: weather_slot = 0
        print("Selected fan powers=",p1," and ",p2, "; Q Process=",q ,"at t ="+str(env.now))
        w_mask = df_weather[weather_slot:(weather_slot+t)]
        Tb = T_basin_calculation(T_basin_initial=Tb_initial,
                                 Q=q,
                                 P_Fan1=p1,
                                 P_Fan2=p2,
                                 weathermask=w_mask)

        noise = np.random.normal(0, 0.05, t*15)
        Tb_noise = Tb + noise

        temp = pd.DataFrame()
        dates = pd.date_range(start=min(w_mask.index) - pd.Timedelta(minutes=15), end=max(w_mask.index)- pd.Timedelta(minutes=1), freq='Min')
        temp['Date_time'] = dates
        temp['Weather_dt'] = w_mask.index.repeat(15)
        temp['Tb'] = Tb
        temp['Tb_noise'] = Tb_noise
        temp['Power_fan_1'] = p1
        temp['Power_fan_2'] = p2
        temp['Q_process'] = q
        data.append(temp)

        Tb_initial = Tb[-1]
        weather_slot = weather_slot+t
        yield env.timeout(t)

if __name__ == '__main__':
    main()











