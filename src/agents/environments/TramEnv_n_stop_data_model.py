import numpy as np
import pandas as pd
from gym import Env
from gym.spaces import Box, Discrete, Dict
from gym.utils import seeding
import random
import math
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers import Dropout
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import pickle

import warnings

VERSION = "1.3 Includes Data Model Energy Predictions"

# Ignore all DeprecationWarnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning)

# TODO: Move all non essential funtions from env to utils. Potentially create multiple utils files if necessary.
# TODO: Add Rainbow DQN

class TramEnv(Env):
    def __init__(self, config: dict, baseline = False):
        super(TramEnv, self).__init__()

        self.seed()
        self.config = config
        self.baseline = baseline

        self.define_vehicle()

        self.hour = 10
        self.temperature = 20
        self.init_position = 5373

        # Data Model
        self.model = tf.keras.models.load_model(r'C:\Users\Admin\OneDrive - thu.de\Dokumente\00_Masterarbeit\01_Code\strassenbahn_nn\Notebooks\MLN_MA5_DBSCAN_MULT_UNIS_5F.keras')
        with open(r'C:\Users\Admin\OneDrive - thu.de\Dokumente\00_Masterarbeit\01_Code\strassenbahn_nn\Notebooks\MLN_MA5_DBSCAN_MULT_UNIS_5F.pkl', 'rb') as f:
            self.scaler = pickle.load(f)

        self.randomize_state = self.config['environment']['randomize_state_init']
        self.random_pos = self.config["environment"]["randomize_position_init"]
        self.max_steps = self.config['environment']['max_steps']
        self.time_steps = self.config['environment']['time_steps']
        self.targets = self.config['environment']['targets']
        self.schedule = self.config['environment']['schedule']

        self.max_line_length = self.config['environment']['max_line_length'] # max length of a track section created in random init

        #self.start_stop = "MULT"#"LEHR"
        #self.end_stop = "UNIS"
        self.start_stop = self.config['environment']['start']
        self.end_stop = self.config['environment']['stop']

        self.load_infra_data()
        self.create_infra(init = True)

        self.max_speed = round(self.config['settings']['vehicle']['max_speed']/3.6)
        self.max_break = -1
        self.max_acc = 1
        self.max_speed_limit = max(self.infra_data["speed_df"]/3.6)

        min_gradient = min(self.infra_data["elevation_df"] / 10)
        max_gradient = max(self.infra_data["elevation_df"] / 10)

        self.min_power = self.calc_power(self.max_speed, self.max_break, min_gradient)
        self.max_power = self.calc_power(self.max_speed, self.max_acc, max_gradient)

        self.low = np.array([
                        0,                                                           # speed
                        0,                                                           # Lower speed limit
                        0,                                                           # current speed limit
                        0,                                                           # Difference to current speed limit
                        0,                                                           # next speed limit 
                        0,                                                           # Difference to next speed limit
                        0,                                                           # distance to next speed limit
                        0,                                                           # Distance traveled in current timestep
                        0,                                                           # next step distance traveled with acc -1
                        0,                                                           # next step distance traveled with acc 0
                        0,                                                           # next step distance traveled with acc 1
                        0,                                                           # distance to next target
                        -self.max_steps,                                           # remaining time to next target
                        min_gradient,                                                # Current elevation
                        self.min_power,                                              # current power consumption
                        self.min_power,                                              # next step power consumption with acc -1
                        self.calc_power(self.max_speed, 0, min_gradient), # next step power consumption with acc 0
                        self.calc_power(1, 1, min_gradient)])             # next step power consumption with acc 1
        

        self.high = np.array([
                        self.max_speed,                                                                 # speed
                        max(self.lower_limit),                                                          # Lower speed limit
                        self.max_speed_limit,                                                           # current speed limit
                        self.max_speed_limit,                                                           # Difference to current speed limit
                        self.max_speed_limit,                                                           # next speed limit 
                        self.max_speed_limit,                                                           # Difference to next speed limit
                        1456,                                                                           # distance to next speed limit
                        0.5 * self.max_acc * self.time_steps ** 2 + self.time_steps * self.max_speed,   # Distance traveled in current timestep
                        0.5 * self.max_break * self.time_steps ** 2 + self.time_steps * self.max_speed, # next step distance traveled with acc -1
                        self.time_steps * self.max_speed,                                               # next step distance traveled with acc 0
                        0.5 * self.max_acc * self.time_steps ** 2 + self.time_steps * self.max_speed,   # next step distance traveled with acc 1
                        self.max_line_length-1,                                                         # distance to next target
                        (self.max_line_length-1)*0.1+40,                                                # remaining time to next target
                        max_gradient,                  # Current elevation
                        self.max_power,                                                                 # current power consumption
                        0,       # next step power consumption with acc -1
                        self.calc_power(self.max_speed, 0, max_gradient),                    # next step power consumption with acc 0
                        self.max_power])                                                                # next step power consumption with acc 1


        self.observation_space = Box(self.low, self.high, dtype=np.float32)

        self.action_space = Discrete(3)

        self.max_step_reward = self.calc_max_step_reward()
        self.reset()

    def define_vehicle(self):
        self.CW = 0.8                       # Luftwiderstandsbeiwert
        self.C_ROLL = 0.000937              # Rollwiderstandsbeiwert
        self.BREITE = 2.4                  # b in m
        self.HÖHE = 3.59                    # h in m
        self.A = self.BREITE * self.HÖHE    # Querschnittsfläche in m^2
        self.LEERGEWICHT = 38000            # m in kg
        self.MAX_ZULADUNG = 13800           # m in kg
        self.MASSENFAKTOR = 1.13            # rotierende masse
        self.P = 1.17                       # Dichte der Luft in kg/m^3
        self.G = 9.81                       # Gewichtskraft in m/s^2

        self.nebenverbraucher = self.config['settings']['vehicle']['auxillary_power']   # P in kW
        self.auslastung = self.config['settings']['vehicle']['passenger_load']          # Fahrzeugauslastung
        self.gesamtgewicht = self.LEERGEWICHT + self.MAX_ZULADUNG*self.auslastung       # Gesamtgewicht in kg
        self.v_wind = self.config['settings']['conditions']['wind_speed']               # Windgeschwindigkeit in m/s

    def load_infra_data(self):
        self.infra_data = pd.read_csv(os.path.dirname(os.path.abspath(__file__)) + "/data/Linie_2_data.csv", sep = ";", decimal = ",")
        self.stop_list = [
            "KUH", "GKOE", "GRIM", "SAAR", "ROEM", 
            "MLUK", "EHI", "HBF", "THEA", "STWE", 
            "LEHR", "MULT", "ESHA", "UNIS", "BOTA", 
            "KLWI", "UNIW", "MBST", "HOCH", "SCIP"]

    def create_infra(self, init = False):
        index_to_start = self.infra_data[self.infra_data['stops_df'] == self.start_stop].index[0]
        index_to_stop =  self.infra_data[self.infra_data['stops_df'] == self.end_stop].index[0]
        df_cut = self.infra_data.loc[index_to_start:index_to_stop].reset_index(drop=True)       

        if len(df_cut) > self.max_line_length:
            df_cut = df_cut[:self.max_line_length]

        self.speed_limits = round(df_cut["speed_df"]/3.6)

        if self.randomize_state and not init:
            self.create_random_targets_schedule()

        if not self.baseline:
            for target in self.targets:
                df_cut.loc[target, "speed_df"] = 0

        self.speed_limits = round(df_cut["speed_df"]/3.6)
        self.speed_limits_positions = df_cut[df_cut['speed_df'] != df_cut['speed_df'].shift(1)].index[1:]
        self.gradient_perc = (df_cut["elevation_df"] / 10).tolist()

        # Define lower speed limits. Reduces to 0 4m befor stop.
        self.lower_limit = [2]*len(self.speed_limits)
        self.lower_limit[:4] = [0,0.5,1,1.5]
        for target in self.targets:
            values_to_assign = [1.5,1,0.5,0,0,0,0,0] if target == max(self.targets) else [1.5,1,0.5,0,0,0,0,0,0.5,1,1.5]
            self.lower_limit[target - 7:target + 4] = values_to_assign

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
    # speed # Lower speed limit # current speed limit # Difference to current speed limit # next speed limit # Difference to next speed limit # distance to next speed limit
    # Distance traveled in current timestep # next step distance traveled with acc -1 # next step distance traveled with acc 0 # next step distance traveled with acc 1
    # distance to next target # remaining time to next target
    # Current elevation 
    # current power consumption # next step power consumption with acc -1 # next step power consumption with acc 0 # next step power consumption with acc 1
        speed = self.state[0]

        self.time += self.time_steps
        self.acceleration = action - 1

        dist_traveled = max(0, 0.5 * self.acceleration * self.time_steps ** 2 + self.time_steps * speed)
        self.position = self.position + dist_traveled
        # Calc Observations
        speed = max(0, self.acceleration * self.time_steps + speed)

        # speed limit info:
        lower_speed_limit = self.lower_limit[min(int(round(self.position)), len(self.lower_limit) - 1)]
        current_speed_limit = self.speed_limits[min(int(round(self.position)), len(self.speed_limits) - 1)]
        dif_to_current_speed_limit = current_speed_limit - speed
        next_speed_limit = self.get_next_speed_limit(self.position) if len(self.speed_limits_positions) > 0 else current_speed_limit
        dif_to_next_speed_limit = next_speed_limit - speed
        if self.baseline:
            dist_next_speed_limit = 0
        else:
            dist_next_speed_limit = min((change - self.position for change in self.speed_limits_positions if change > self.position), default=0)

        # next distance info:
        next_dist_traveled_break = max(0, 0.5 * self.max_break * self.time_steps ** 2 + self.time_steps * speed)
        next_dist_traveled_cruise = max(0, self.time_steps * speed)
        next_dist_traveled_acc = max(0, 0.5 * self.max_acc * self.time_steps ** 2 + self.time_steps * speed)

        # calc remaining distance to next target:
        dist_target = min((target - self.position for target in self.targets if target > self.position), default=0)
        #schedule update:
        remaining_time = self.calc_remaining_time(self.position)

        elevation = 0
        if self.config['rewards']['energy']['step']['power']['use_track_gradient']:
            elevation =  self.gradient_perc[min(int(round(self.position)), len(self.gradient_perc) - 1)]

        power = self.pred_power(speed, self.position, self.acceleration, elevation)
        next_power_break = self.calc_power(max(0,speed+self.max_break), self.max_break, self.gradient_perc[min(int(self.position+next_dist_traveled_break), len(self.gradient_perc) - 1)])
        next_power_cruise = self.calc_power(speed, 0, self.gradient_perc[min(int(self.position+next_dist_traveled_cruise), len(self.gradient_perc) - 1)])
        next_power_acc = self.calc_power(speed+self.max_acc, self.max_acc, self.gradient_perc[min(int(self.position+next_dist_traveled_acc), len(self.gradient_perc) - 1)])

        self.state = (speed, lower_speed_limit, current_speed_limit, dif_to_current_speed_limit, 
                      next_speed_limit, dif_to_next_speed_limit, 
                      dist_next_speed_limit, 
                            dist_traveled, next_dist_traveled_break, next_dist_traveled_cruise, next_dist_traveled_acc, dist_target, remaining_time, 
                            elevation, power, next_power_break, next_power_cruise, next_power_acc)
        
        reward = self.calc_reward(speed, lower_speed_limit, current_speed_limit, remaining_time, power)
        done = self.max_steps <= self.time / self.time_steps or self.position >= max(self.targets) #or speed > self.max_speed

        info = {#"State": self.state,
                "total_reward":reward,
                "speed_reward":self.vel_reward,
                "pos_reward": self.pos_reward,
                "schedule_reward": self.sched_reward,
                "energy_reward": self.energy_reward,
                "cumsum_energy_consumption": self.cumsum_energy, 
                "pos_return": self.pos_return,
                "vel_return": self.vel_return,
                "sched_return": self.sched_return,
                "energy_return": self.energy_return}

        return np.array(self.scale_state(self.state), dtype=np.float32), self.scale_reward(reward, max_value=self.max_step_reward), done, info

    def calc_reward(self, speed, lower_speed_limit, current_speed_limit, remaining_time, power):
        self.pos_reward = self.positional_rewards(speed)
        self.vel_reward = self.velocity_rewards(speed, lower_speed_limit, current_speed_limit)
        self.sched_reward = self.schedule_rewards(remaining_time)
        self.energy_reward = self.energy_rewards(power)

        self.pos_return += self.pos_reward
        self.vel_return += self.vel_reward
        self.sched_return += self.sched_reward
        self.energy_return += self.energy_reward

        reward = self.pos_reward + self.vel_reward + self.sched_reward + self.energy_reward

        return reward

    def positional_rewards(self, speed):
        reward = 0
        if self.config['rewards']['positional']['use_stopZone']:
            stop_area = self.config['rewards']['positional']['stop_zone']['value']

            if self.config['rewards']['positional']['use_quadratic_rewards']:
                full_reward = self.config['rewards']['positional']['value']
                for i, target in enumerate(self.targets):
                    if abs(self.position - target) <= stop_area and speed == 0 and not self.rewards_achieved[i]:    
                        reward += -(1/(stop_area+1)**2)*full_reward*(self.position - target)**2 + full_reward                
                        self.rewards_achieved[i] = True

            else:
                for i, target in enumerate(self.targets):
                    if self.position == target and not self.rewards_achieved[i]:
                        reward += 1/(speed+1) * self.config['rewards']['positional']['value']     
                        self.rewards_achieved[i] = True                

        else:
            for i, target in enumerate(self.targets):
                if self.position == target and not self.rewards_achieved[i]:

                    reward += 1/(speed+1) * self.config['rewards']['positional']['value']     
                    self.rewards_achieved[i] = True                   

        return reward

    def velocity_rewards(self, speed, lower_speed_limit, current_speed_limit):
        reward = 0
        if speed >= lower_speed_limit and speed <= current_speed_limit:
            reward+= self.config['rewards']['velocity']['speed_reward']
        
        elif speed > current_speed_limit:
            reward -= abs(speed-current_speed_limit)*self.config['rewards']['velocity']['speeding_cost']
        
        elif speed < lower_speed_limit:
            reward -= abs(speed-lower_speed_limit)*self.config['rewards']['velocity']['slow_cost']
        
        return reward
    
    def schedule_rewards(self, remaining_time):       
        reward = 0
        
        # Step Rewards
        if self.config['rewards']['schedule']['use_step']:
            if remaining_time < 0:
                reward+= remaining_time*self.config['rewards']['schedule']['step']['value']
        
        # Sparse Rewards
        if self.config['rewards']['schedule']['use_sparse']:
            for target in self.targets:
                if self.position == target and remaining_time < 0:
                    reward+= remaining_time * self.config['rewards']['schedule']['sparse']['value']
        
        return reward        

    def energy_rewards(self, power):
        reward = 0
        #TODO: Implement sparse rewards
        self.cumsum_energy += max(0, power/3600)
        
        if self.config['rewards']['energy']['use_step']:
            reward+= ((power if power > 0 else 0)/self.config['rewards']['energy']['step']['power']['value'])*-1 

        return reward

    def calc_max_step_reward(self):
        #lowest_power_val = self.calc_power(self.max_speed,self.max_break,min(self.gradient_perc))
        highest_power_val = self.calc_power(self.max_speed,self.max_acc,max(self.gradient_perc))

        max_power_reward = 0
        min_power_reward = 0
        if self.config['rewards']['energy']['use_step']:
            #max_power_reward = (lowest_power_val/self.config['rewards']['energy']['step']['power']['value'])*-1 
            min_power_reward = (highest_power_val/self.config['rewards']['energy']['step']['power']['value'])*-1

        rewards = self.config['rewards']
        max_reward = rewards['positional']['value'] + rewards['velocity']['speed_reward'] + max_power_reward
        
        sched = 0 
        if rewards['schedule']['use_sparse']:
            sched += rewards['schedule']['sparse']['value'] * (self.max_steps-max(self.schedule)-1)
        if rewards['schedule']['use_step']:
            sched += rewards['schedule']['step']['value'] * (self.max_steps-max(self.schedule)-1)

        min_reward = rewards['velocity']['speeding_cost'] + rewards['velocity']['slow_cost'] + sched + abs(min_power_reward)

        return max(min_reward, max_reward)

    def calc_remaining_time(self,position):
        for i in range(len(self.section_list )-1):
            if self.section_list [i] <= position <= self.section_list [i + 1]:
                self.remaining_time[i]-=1
                time_left = self.remaining_time[i]
            elif position > self.section_list[-1]:
                self.remaining_time[-1]-=1
                time_left = self.remaining_time[-1]
                break
        return time_left    

    def get_next_speed_limit(self, position):
        valid_values = [value for value in self.speed_limits_positions if value > position]
        if valid_values:
            closest_value = min(valid_values, key=lambda x: abs(x - position))
        else:
            closest_value = self.speed_limits_positions[-1] if len(self.speed_limits_positions) > 0 else None

        return self.speed_limits[closest_value]

    def calc_power(self, speed, acceleration, gradient):
        v_rel = speed+self.v_wind
        steigung_grad = math.degrees(math.atan(gradient / 100))

        luftwiderstand_spez = (self.CW*self.A*self.P*v_rel**2)/(2*self.gesamtgewicht*self.G)
        rollwiderstand_spez = self.C_ROLL * math.cos(math.radians(steigung_grad))  
        steigungswiderstand_spez = math.sin(math.radians(steigung_grad))
        beschleunigungswiderstand_spez = (acceleration*self.MASSENFAKTOR)/self.G
        krümmungswiderstand_spez = 0 # TODO: Implement Krümmungswiderstand

        Zugkraft = self.gesamtgewicht * self.G * (luftwiderstand_spez+
                                        rollwiderstand_spez+
                                        steigungswiderstand_spez+
                                        beschleunigungswiderstand_spez+
                                        krümmungswiderstand_spez)

        Leistung = (Zugkraft * speed)/1000
        Leistung_Gesamt = Leistung + self.nebenverbraucher
        return Leistung_Gesamt

    def pred_power(self, speed, position, acceleration, elevation):
        #Speed [kmh]	Temperatur	Position elevation	daytimeHour	acceleration
        
        position = position + self.init_position

        input_data = np.array([speed,self.temperature,position,elevation,self.hour,acceleration])	

        input_data_scaled = self.scaler.transform(input_data.reshape(1, -1))  # Scale the input data
        prediction = self.model.predict(input_data_scaled)

        return prediction[0][0]

    def scale_state(self, state):
        return tuple((np.array(state) - self.low) / (self.high - self.low))
        # TODO: add logarithmic scaling
        # (log(epsilon + x) - log(epsilon)) / log(epsilon)   
        # epsilon z.b. 0.0004

    def scale_reward(self, reward, max_value):
        scaled_reward = reward/max_value
        return round(scaled_reward,5)     

    def render(self, mode):
        raise NotImplementedError("This method is not yet implemented.")

    def get_state(self):
        return self.state

    def reset(self):
        self.time = 0  # time in s
        self.position = 0
        
        if self.randomize_state and not self.baseline:
            self.create_random_track_section()
            self.create_infra()
            if self.random_pos:
                self.position, speed, idx = self.create_random_pos_speed()
            self.max_step_reward = self.calc_max_step_reward()

        self.section_list = [0]+self.targets



        if self.randomize_state and not self.baseline and self.random_pos:
            next_dist_traveled_break = max(0, 0.5 * self.max_break * self.time_steps ** 2 + self.time_steps * speed)
            next_dist_traveled_cruise = max(0, self.time_steps * speed)
            next_dist_traveled_acc = max(0, 0.5 * self.max_acc * self.time_steps ** 2 + self.time_steps * speed)

            self.state = np.array([ 
                                speed,                                              #speed
                                self.lower_limit[self.position],                    #Lower speed limit
                                self.speed_limits[self.position],                   #current speed limit
                                self.speed_limits[self.position] - speed,           #Difference to current speed limit
                                self.get_next_speed_limit(self.position),           #next speed limit
                                self.get_next_speed_limit(self.position) - speed,   #Difference to next speed limit
                                min((x - self.position for x in self.speed_limits_positions if x > self.position), default=0), #distance to next speed limit
                                max(0, self.time_steps * speed),                    # Distance traveled in current timestep
                                next_dist_traveled_break,                           # next step distance traveled with acc -1
                                next_dist_traveled_cruise,                          # next step distance traveled with acc 0
                                next_dist_traveled_acc,                             # next step distance traveled with acc 1
                                min((target - self.position for target in self.targets if target > self.position), default=0),                                          # distance to next target
                                self.schedule[idx],                                 # remaining time to next target
                                self.gradient_perc[self.position] if self.config['rewards']['energy']['step']['power']['use_track_gradient'] else 0,                    # Current elevation
                                self.pred_power(speed,
                                                self.position,
                                                0,
                                                self.gradient_perc[self.position] if self.config['rewards']['energy']['step']['power']['use_track_gradient'] else 0),   # current power consumption
                                self.pred_power(speed+self.max_break,
                                                self.position+next_dist_traveled_break,
                                                self.max_break,
                                                self.gradient_perc[min(int(round(self.position+next_dist_traveled_break)), len(self.gradient_perc) - 1)] if self.config['rewards']['energy']['step']['power']['use_track_gradient'] else 0),                                                                                                  # next step power consumption with acc -1
                                self.pred_power(speed,
                                                self.position+next_dist_traveled_cruise,
                                                0,
                                                self.gradient_perc[min(int(round(self.position+next_dist_traveled_cruise)), len(self.gradient_perc) - 1)] if self.config['rewards']['energy']['step']['power']['use_track_gradient'] else 0),# next step power consumption with acc 0
                                self.pred_power(speed+self.max_acc,
                                                self.position+next_dist_traveled_acc,
                                                self.max_acc,
                                                self.gradient_perc[min(int(round(self.position+next_dist_traveled_acc)), len(self.gradient_perc) - 1)] if self.config['rewards']['energy']['step']['power']['use_track_gradient'] else 0)# next step power consumption with acc 1
                                ]) 
        else:
            self.state = np.array([
                                0,                                                                                                                      # speed 
                                0,                                                                                                                      # Lower speed limit
                                self.speed_limits[0],                                                                                                   # current speed limit
                                self.speed_limits[0],                                                                                                   # Difference to current speed limit
                                self.speed_limits[self.speed_limits_positions[0]] if len(self.speed_limits_positions) > 0 else self.speed_limits[0],    # next speed limit 
                                self.speed_limits[self.speed_limits_positions[0]] if len(self.speed_limits_positions) > 0 else self.speed_limits[0],    # Difference to next speed limit
                                self.speed_limits_positions[0] if len(self.speed_limits_positions) > 0 else 0,                                          # distance to next speed limit
                                0,                                                                                                                      # Distance traveled in current timestep
                                0,                                                                                                                      # next step distance traveled with acc -1
                                0,                                                                                                                      # next step distance traveled with acc 0
                                0.5,                                                                                                                    # next step distance traveled with acc 1
                                self.targets[0],                                                                                                        # distance to next target
                                self.schedule[0],                                                                                                       # remaining time to next target
                                self.gradient_perc[0] if self.config['rewards']['energy']['step']['power']['use_track_gradient'] else 0,                # Current elevation
                                0,                                                                                                                      # current power consumption
                                self.nebenverbraucher,                                                                                                  # next step power consumption with acc -1
                                self.nebenverbraucher,                                                                                                  # next step power consumption with acc 0
                                self.pred_power(1,
                                                self.position,
                                                self.max_acc,
                                                self.gradient_perc[0] if self.config['rewards']['energy']['step']['power']['use_track_gradient'] else 0)# next step power consumption with acc 1
                                ])

        self.rewards_achieved = [False] * len(self.targets)
        self.remaining_time = self.schedule.copy()
        self.cumsum_energy = 0

        self.pos_return = 0
        self.vel_return = 0
        self.sched_return = 0
        self.energy_return = 0

        return np.array(self.scale_state(self.state), dtype=np.float32)

    def create_random_targets_schedule(self):
        random_targets = []
        line_length = min(self.max_line_length, len(self.speed_limits))
        for x in range(len(self.targets)):
            random_targets.append(random.randint(int(x * line_length/len(self.targets)), 
                                                 int((x+1) * line_length/len(self.targets) -1)))  
              
        random_schedule = [int(random_targets[0]*0.1 + random.randint(0,40))]
        for x in range(1,len(random_targets)):
            random_schedule.append(int((random_targets[x]-random_targets[x-1])*0.1 + random.randint(0,40)))
        
        self.targets = random_targets
        self.schedule = random_schedule
    
    def create_random_track_section(self):
        first_stop_index = random.randint(0, len(self.stop_list) - 2)
        self.start_stop = self.stop_list[first_stop_index]
        remaining_stops = self.stop_list[first_stop_index + 1:]
        self.end_stop = random.choice(remaining_stops)

    def create_random_pos_speed(self):
        ''' 
        Inits a random position between 0 and max(self.targets), 
        Inits a random speed between 0 and the max possible speed to still be able to break to 0 at the next target  
        Adjusts the schedule for the current segment 

        Inputs
        ------
        None

        Outputs
        -------
        position (int): Randomly generated position
        max speed (int): Randomly generated position based on max speed possible
        idx (int): Index of the next target in self.targets based on generated position
        '''
        position = random.randint(0,max(self.targets))
        idx, closest_target = [(index, target) for index, target in enumerate(self.targets) if target >= position][0]
        dist_target = closest_target - position
        self.schedule[idx] = int(dist_target*0.1) + random.randint(0,40)
        max_break = abs(self.max_break)
        current_speed_limit = int(self.speed_limits[min(int(round(position)), len(self.speed_limits) - 1)])

        for x in range(current_speed_limit+1):
            stopping_distance = (x ** 2) / (2 * max_break)
            if stopping_distance > dist_target:
                x-=1
                break

        return position, random.randint(0,x), idx