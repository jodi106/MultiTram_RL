import numpy as np
import pandas as pd
from gym import Env
from gym.spaces import Box, Discrete, Dict, MultiDiscrete
from gym.utils import seeding
import random
import math
import os
from itertools import product

import warnings

VERSION = "1.4 Speed Reward, Random Schedule Calc Update"

# Ignore all DeprecationWarnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning)

# TODO: Move all non essential funtions from env to utils. Potentially create multiple utils files if necessary.
# TODO: Implement more than two start positions for agent eval if n_trams > 2

class TramEnv(Env):
    def __init__(self, config: dict, baseline = False):
        super(TramEnv, self).__init__()

        self.episode_counter = -1
        self.do_random_test = False

        self.seed()
        self.config = config
        self.baseline = baseline
        self.n_vehicles = self.config['environment']['n_vehicles']
        self.n_actions = 3

        self.define_vehicle()

        self.randomize_state = self.config['environment']['randomize_state_init']
        self.det_random_test = self.config['environment']['det_random_test']
        self.random_test_intervall = self.config['environment']['random_test_intervall']
        self.random_pos = self.config["environment"]["randomize_position_init"]
        self.max_steps = self.config['environment']['max_steps']
        self.time_steps = self.config['environment']['time_steps']


        self.targets = self.config['environment']['targets']
        self.reverse_targets = self.targets[::-1][1:] + [0]

        self.schedule = self.config['environment']['schedule']
        self.reverse_schedule = self.schedule[::-1]

        self.max_line_length = self.config['environment']['max_line_length'] # max length of a track section created in random init

        self.start_stop = self.config['environment']['start']
        self.end_stop = self.config['environment']['stop']
        self.load_infra_data()
        self.create_infra(init = True)

        self.max_speed = round(self.config['settings']['vehicle']['max_speed']/3.6)
        self.max_break = -1
        self.max_acc = 1
        self.max_speed_limit = max(self.infra_data["speed_df"]/3.6)

        min_gradient = min(min(self.infra_data["elevation_df"] / 10), min(self.infra_data["elevation_df"] / -10))
        max_gradient = max(max(self.infra_data["elevation_df"] / 10), max(self.infra_data["elevation_df"] / -10))

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
                        min_gradient,                                     # Current elevation
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
                        max_gradient,                                                        # Current elevation
                        self.max_power,                                                                 # current power consumption
                        0,       # next step power consumption with acc -1
                        self.calc_power(self.max_speed, 0, max_gradient),                    # next step power consumption with acc 0
                        self.max_power])          

        
        self.n_tram_obs = len(self.high)

        self.low_n = np.tile(self.low, self.n_vehicles)
        self.low_n = np.append(self.low_n, 0)               # dissipated Power        

        self.high_n = np.tile(self.high, self.n_vehicles)
        self.high_n = np.append(self.high_n, -1*self.calc_power(self.max_speed, self.max_break, min(self.infra_data['elevation_df']/10)))         # dissipated Power

        self.observation_space = Box(self.low_n, self.high_n, dtype=np.float32)

        self.action_space = MultiDiscrete([self.n_actions]*self.n_vehicles)

        self.max_step_reward = self.calc_max_step_reward()

        keys = ["Speed", "Low_Speed", "Curr_Speed_Limit", "Diff_To_Curr_Speed_Limit", "Next_Speed_Limit", "Diff_To_Next_Speed_Limit", "Dist_To_Next_Speed_Limit", 
                "Dist_Traveled", "Next_Dist_Travaled-1", "Next_Dist_Travaled0", "Next_Dist_Travaled+1", "Dist_Target", "Remain_Time", "Elevation", "Power", "Next_Power-1", "Next_Power0", "Next_Power1"]
        self.keys_extend = []
        for x in range(self.n_vehicles):
            self.keys_extend+= [f"{key}_{x}" for key in keys]
        self.keys_extend.append("Recup_Power")#("Diss_Power")  

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
            self.reverse_targets = self.targets[::-1][1:] + [0]
            self.reverse_schedule = self.schedule[::-1]


        if not self.baseline:
            for target in self.targets[:-1]:
                df_cut.loc[target, "speed_df"] = 0

        df_rev = df_cut.copy()
        df_cut.loc[max(self.targets), "speed_df"] = 0
        self.speed_limits = round(df_cut["speed_df"]/3.6)
        self.speed_limits_positions = df_cut[df_cut['speed_df'] != df_cut['speed_df'].shift(1)].index[1:]

        df_rev = df_rev[::-1]
        df_rev["speed_df"].iloc[-1] = 0
        self.reverse_speed_limits = round(df_rev["speed_df"]/3.6)
        self.reverse_speed_limits_positions = pd.Index(list(df_rev[df_rev['speed_df'] != df_rev['speed_df'].shift(1)].index[2:] ) + [0], dtype='int64')


        self.gradient_perc = (df_cut["elevation_df"] / 10).tolist()

        # Define lower speed limits. Reduces to 0 4m befor stop.
        self.lower_limit = [2]*len(self.speed_limits)
        self.lower_limit[:4] = [0,0.5,1,1.5]
        for target in self.targets:
            values_to_assign = [1.5,1,0.5,0,0,0,0,0] if target == max(self.targets) else [1.5,1,0.5,0,0,0,0,0,0,0,0,0,0.5,1,1.5]
            self.lower_limit[target - 7:target + 8] = values_to_assign

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, actions):
        list_state = list(self.state)

        self.time += self.time_steps
        if isinstance(actions, int):
            actions = [actions]
        self.accelerations = [x - 1 for x in actions]

        speeds = [self.state[x*self.n_tram_obs] for x in range(self.n_vehicles)]
        dist_traveleds = [0]*self.n_vehicles

        for i, speed in enumerate(speeds):
            dist_traveleds[i] = max(0, 0.5 * self.accelerations[i] * self.time_steps ** 2 + self.time_steps * speed)
            speeds[i] = max(0, self.accelerations[i] * self.time_steps + speed) if not self.rewards_achieved_list[i][-1] else 0
        # ---- NEW OBS -------
        # speed limit info:
        lower_speed_limits = [0]*self.n_vehicles
        current_speed_limits = [0]*self.n_vehicles
        dif_to_current_speed_limits = [0]*self.n_vehicles
        next_speed_limits = [0]*self.n_vehicles
        dif_to_next_speed_limits = [0]*self.n_vehicles
        dist_next_speed_limits = [0]*self.n_vehicles

        # next distance info:
        next_dist_traveled_breaks = [0]*self.n_vehicles
        next_dist_traveled_cruises = [0]*self.n_vehicles
        next_dist_traveled_accs = [0]*self.n_vehicles

        # remaining distance to next target:
        dist_targets = [0]*self.n_vehicles
        # schedule update:
        remaining_times = [0]*self.n_vehicles
        
        # power info
        elevations = [0]*self.n_vehicles
        powers = [0]*self.n_vehicles
        next_power_breaks = [0]*self.n_vehicles
        next_power_cruises = [0]*self.n_vehicles
        next_power_accs = [0]*self.n_vehicles
        
        # calc state
        for i, position in enumerate(self.positions):
            if i % 2 == 0:
                self.positions[i] = position + dist_traveleds[i] if not self.rewards_achieved_list[i][-1] else position
                if not self.rewards_achieved_list[i][-1]:
                    lower_speed_limits[i] = self.lower_limit[min(int(round(self.positions[i])), len(self.lower_limit) - 1)]
                    current_speed_limits[i] = self.speed_limits[min(int(round(self.positions[i])), len(self.speed_limits) - 1)]
                    next_speed_limits[i] = self.get_next_speed_limit(self.positions[i])
                    dist_next_speed_limits[i] = min((change - self.positions[i] for change in self.speed_limits_positions if change > self.positions[i]), default=0)
                    dist_targets[i] = min((target - self.positions[i] for target in self.targets if target > self.positions[i]), default=0)
                    remaining_times[i] = self.calc_remaining_time(self.positions[i],tram_index=i)
                    if self.config['rewards']['energy']['step']['power']['use_track_gradient']:
                        elevations[i] =  self.gradient_perc[min(int(round(self.positions[i])), len(self.gradient_perc) - 1)]
                    next_power_breaks[i] = self.calc_power(speeds[i]+self.max_break, 
                                                        self.max_break, 
                                                        self.gradient_perc[min(int(round(self.positions[i]+next_dist_traveled_breaks[i])), len(self.gradient_perc) - 1)])
                    next_power_cruises[i] = self.calc_power(speeds[i], 
                                                            0, 
                                                            self.gradient_perc[min(int(round(self.positions[i]+next_dist_traveled_cruises[i])), len(self.gradient_perc) - 1)])
                    next_power_accs[i] = self.calc_power(speeds[i]+self.max_acc, 
                                                        self.max_acc, 
                                                        self.gradient_perc[min(int(round(self.positions[i]+next_dist_traveled_accs[i])), len(self.gradient_perc) - 1)])
            else:
                self.positions[i] = position - dist_traveleds[i] if not self.rewards_achieved_list[i][-1] else position
                if not self.rewards_achieved_list[i][-1]:
                    lower_speed_limits[i] = self.lower_limit[max(int(round(self.positions[i])), 0)]
                    current_speed_limits[i] = self.reverse_speed_limits[max(int(round(self.positions[i])), 0)]
                    next_speed_limits[i] = self.get_next_speed_limit(self.positions[i], reverse = True)
                    dist_next_speed_limits[i] = min((self.positions[i] - change for change in self.reverse_speed_limits_positions if change < self.positions[i]), default=0)
                    dist_targets[i] = min((self.positions[i] - target  for target in self.reverse_targets if target < self.positions[i]), default=0)
                    remaining_times[i] = self.calc_remaining_time(self.positions[i], tram_index=i, reverse=True)
                    if self.config['rewards']['energy']['step']['power']['use_track_gradient']:
                        elevations[i] =  -self.gradient_perc[max(int(round(self.positions[i])), 0)]         
                    next_power_breaks[i] = self.calc_power(speeds[i]+self.max_break, 
                                                        self.max_break, 
                                                        -self.gradient_perc[max(int(round(self.positions[i]-next_dist_traveled_breaks[i])), 0)])
                    next_power_cruises[i] = self.calc_power(speeds[i], 
                                                            0, 
                                                            -self.gradient_perc[max(int(round(self.positions[i]-next_dist_traveled_cruises[i])), 0)])
                    next_power_accs[i] = self.calc_power(speeds[i]+self.max_acc, 
                                                        self.max_acc, 
                                                        -self.gradient_perc[max(int(round(self.positions[i]-next_dist_traveled_accs[i])), 0)])

            if not self.rewards_achieved_list[i][-1]:
                dif_to_current_speed_limits[i] = current_speed_limits[i] - speeds[i]
                dif_to_next_speed_limits[i] = next_speed_limits[i] - speeds[i]

                next_dist_traveled_breaks[i] = max(0, 0.5 * self.max_break * self.time_steps ** 2 + self.time_steps * speeds[i])
                next_dist_traveled_cruises[i] = max(0, self.time_steps * speeds[i])
                next_dist_traveled_accs[i] = max(0, 0.5 * self.max_acc * self.time_steps ** 2 + self.time_steps * speeds[i])

                powers[i] = self.calc_power(speeds[i], self.accelerations[i], elevations[i])

        for x in range(self.n_vehicles):
            index = x * self.n_tram_obs
            list_state[index:index+self.n_tram_obs] = (
                                                speeds[x], 
                                                lower_speed_limits[x], 
                                                current_speed_limits[x], 
                                                dif_to_current_speed_limits[x], 
                                                next_speed_limits[x], 
                                                dif_to_next_speed_limits[x], 
                                                dist_next_speed_limits[x],
                                                dist_traveleds[x],
                                                next_dist_traveled_breaks[x],
                                                next_dist_traveled_cruises[x],
                                                next_dist_traveled_accs[x],
                                                dist_targets[x],
                                                remaining_times[x],
                                                elevations[x],
                                                powers[x],
                                                next_power_breaks[x],
                                                next_power_cruises[x],
                                                next_power_accs[x])

        reward = self.calc_reward(speeds, lower_speed_limits, current_speed_limits, remaining_times, powers)
        # To avoid redundant function calling, dissipated energy is calculated within calc_reward
        list_state[-1] = self.recuperation_power #self.dissipated_power
        self.state = tuple(list_state)

        # -- Calculate done -- #
        odd_positions_satisfy_condition = all(
            position <= min(self.reverse_targets)
            for i, position in enumerate(self.positions)
            if i % 2 == 1
        )

        even_positions_satisfy_condition = all(
            position >= max(self.targets)
            for i, position in enumerate(self.positions)
            if i % 2 == 0 or i == 0
        )
        done =  (
            self.max_steps <= self.time / self.time_steps 
            or (odd_positions_satisfy_condition and even_positions_satisfy_condition) 
            or all(x[-1] for x in self.rewards_achieved_list)
        )


        # -- Create State dict for info -- # 
        result_dict = {key: value for key, value in zip(self.keys_extend, self.state)}
        result_dict["Positions"] = self.positions.copy()
        
        info = {"State": result_dict,
                "total_reward":reward,
                "speed_reward":self.vel_rewards,
                "pos_reward": self.pos_rewards,
                "schedule_reward": self.sched_rewards,
                "energy_reward": self.power_rewards+self.recup_reward,
                "cumsum_energy_consumption": self.cumsum_energy, 
                "pos_return": self.pos_return,
                "vel_return": self.vel_return,
                "sched_return": self.sched_return,
                "energy_return": self.power_return,
                "recuperated_power": self.recuperation_power,
                "dissipated_power": self.dissipated_power,
                "episode_counter": self.episode_counter,
                "det_test_random": self.do_random_test}

        return np.array(self.scale_state(self.state), dtype=np.float32), self.scale_reward(reward, max_value=self.max_step_reward), done, info

    def calc_reward(self, speeds, lower_speed_limits, current_speed_limits, remaining_times, powers):

        self.pos_rewards = sum([self.positional_rewards(position,speed, x) for x, (position, speed) in enumerate(zip(self.positions, speeds))])
        self.vel_rewards = sum([self.velocity_rewards(speed, current_speed_limit, lower_speed_limit, x) for x, (speed, 
                                current_speed_limit, lower_speed_limit) in enumerate(zip(speeds, current_speed_limits, lower_speed_limits))])
        self.sched_rewards = sum([self.schedule_rewards(position, remaining_time, x) for x, 
                                  (position, remaining_time) in enumerate(zip(self.positions, remaining_times))])
        self.power_rewards = sum([self.energy_rewards(power,x) for x, power in enumerate(powers)])
        
        # Calc recuperated/ dissipated power
        self.recup_reward = self.recuperation_rewards(powers)  

        self.pos_return += self.pos_rewards
        self.vel_return += self.vel_rewards
        self.sched_return += self.sched_rewards
        self.power_return += (self.power_rewards+self.recup_reward)

        reward = self.pos_rewards + self.vel_rewards + self.sched_rewards + self.power_rewards + self.recup_reward

        return reward

    def positional_rewards(self, position, speed, ind = 0):
        reward = 0
        if not self.rewards_achieved_list[ind][-1]:
            targets = self.targets
            if ind % 2:
                targets = self.reverse_targets
                
            if self.config['rewards']['positional']['use_stopZone']:
                stop_area = self.config['rewards']['positional']['stop_zone']['value']

                if self.config['rewards']['positional']['use_quadratic_rewards']:
                    full_reward = self.config['rewards']['positional']['value']
                    for i, target in enumerate(targets):
                        if abs(position - target) <= stop_area and speed == 0 and not self.rewards_achieved_list[ind][i]:    
                            reward += -(1/(stop_area+1)**2)*full_reward*(position - target)**2 + full_reward                
                            self.rewards_achieved_list[ind][i] = True

                else:
                    for i, target in enumerate(targets):
                        if position == target and not self.rewards_achieved_list[ind][i]:
                            reward += 1/(speed+1) * self.config['rewards']['positional']['value']     
                            self.rewards_achieved_list[ind][i] = True                

            else:
                for i, target in enumerate(targets):
                    if position == target and not self.rewards_achieved_list[ind][i]:

                        reward += 1/(speed+1) * self.config['rewards']['positional']['value']     
                        self.rewards_achieved_list[ind][i] = True                   

        return reward

    def velocity_rewards(self, speed, current_speed_limit, lower_speed_limit, ind):
        reward = 0
        if not self.rewards_achieved_list[ind][-1]:
            if speed >= lower_speed_limit and speed <= current_speed_limit:
                if lower_speed_limit > 0:
                    reward+= self.config['rewards']['velocity']['speed_reward']
            
            elif speed > current_speed_limit:
                reward -= abs(speed-current_speed_limit)*self.config['rewards']['velocity']['speeding_cost']
            
            elif speed < lower_speed_limit:
                reward -= abs(speed-lower_speed_limit)*self.config['rewards']['velocity']['slow_cost']
        
        return reward
    
    def schedule_rewards(self, position, remaining_time, ind = 0):       
        reward = 0
        if not self.rewards_achieved_list[ind][-1]:
            targets = self.targets
            if ind % 2:
                targets = self.reverse_targets
            # Step Rewards
            if self.config['rewards']['schedule']['use_step']:
                if remaining_time < 0:
                    reward+= remaining_time*self.config['rewards']['schedule']['step']['value']
            
            # Sparse Rewards
            if self.config['rewards']['schedule']['use_sparse']:
                for target in targets:
                    if position == target and remaining_time < 0:
                        reward+= remaining_time * self.config['rewards']['schedule']['sparse']['value']
        
        return reward        

    def energy_rewards(self, power, ind):
        reward = 0

        if not self.rewards_achieved_list[ind][-1]:
            #TODO: Implement sparse rewards
            self.cumsum_energy += max(0, power/3600)
            
            if self.config['rewards']['energy']['use_step']:
                reward+= ((power if power > 0 else 0)/self.config['rewards']['energy']['step']['power']['value'])*-1 

        return reward
    
    def recuperation_rewards(self, powers):
        if sum(powers) >= 0:
            self.recuperation_power = abs(sum(x for x in powers if x < 0))
            self.dissipated_power = 0
        elif sum(powers) < 0:
            self.recuperation_power = sum(x for x in powers if x > 0)
            self.dissipated_power = abs(sum(x for x in powers if x < 0)) - self.recuperation_power

        return (self.recuperation_power/self.config['rewards']['energy']['step']['power']['value'])*self.config['rewards']['energy']['recup_factor']

    def calc_max_step_reward(self):
        lowest_power_val = self.min_power
        highest_power_val = self.max_power

        max_power_reward = 0
        min_power_reward = 0
        if self.config['rewards']['energy']['use_step']:
            max_power_reward = (lowest_power_val/self.config['rewards']['energy']['step']['power']['value'])*-1 
            min_power_reward = (highest_power_val/self.config['rewards']['energy']['step']['power']['value'])*-1

        rewards = self.config['rewards']
        max_reward = (rewards['positional']['value'] + rewards['velocity']['speed_reward']) * self.n_vehicles + max_power_reward * (self.n_vehicles-1)
        
        sched = 0 
        if rewards['schedule']['use_sparse']:
            sched += rewards['schedule']['sparse']['value'] * (self.max_steps-max(self.schedule)-1)
        if rewards['schedule']['use_step']:
            sched += rewards['schedule']['step']['value'] * (self.max_steps-max(self.schedule)-1)

        min_reward = (rewards['velocity']['speeding_cost'] + rewards['velocity']['slow_cost'] + sched + abs(min_power_reward)) * self.n_vehicles

        return max(min_reward, max_reward)

    def calc_remaining_time(self, position, tram_index, reverse=False):
        # Check for reverse direction
        if reverse:
            for i in range(len(self.reverse_section_list) - 1):
                if self.reverse_section_list[i] >= position >= self.reverse_section_list[i + 1]:
                    # Decrement only the remaining time specific to the current tram
                    self.reverse_remaining_time_per_tram[tram_index][i] -= 1
                    time_left = self.reverse_remaining_time_per_tram[tram_index][i]
                    break
            else:
                # Handle case where position is beyond the last section
                if position < self.reverse_section_list[-1]:
                    self.reverse_remaining_time_per_tram[tram_index][-1] -= 1
                    time_left = self.reverse_remaining_time_per_tram[tram_index][-1]

        # Check for forward direction
        else:
            for i in range(len(self.section_list) - 1):
                if self.section_list[i] <= position <= self.section_list[i + 1]:
                    # Decrement only the remaining time specific to the current tram
                    self.remaining_time_per_tram[tram_index][i] -= 1
                    time_left = self.remaining_time_per_tram[tram_index][i]
                    break
            else:
                # Handle case where position is beyond the last section
                if position > self.section_list[-1]:
                    self.remaining_time_per_tram[tram_index][-1] -= 1
                    time_left = self.remaining_time_per_tram[tram_index][-1]

        return time_left


    def get_next_speed_limit(self, position, reverse=False):
        index_val = -1
        valid_values = [value for value in self.speed_limits_positions if value > position]
        if reverse:
            index_val = 0
            valid_values = [value for value in self.reverse_speed_limits_positions if value < position]
        if valid_values:
            closest_value = min(valid_values, key=lambda x: abs(x - position))
        else:
            closest_value = self.speed_limits_positions[index_val] if len(self.speed_limits_positions) > 0 else None

        if reverse:
            return self.reverse_speed_limits[closest_value]
        else:
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

    def scale_state(self, state):
        return tuple((state - self.low_n) / (self.high_n - self.low_n))     

    def scale_reward(self, reward, max_value):
        scaled_reward = reward/max_value
        return round(scaled_reward,5)     

    def render(self, mode):
        raise NotImplementedError("This method is not yet implemented.")

    def get_state(self):
        return self.state

    def reset(self):
        self.episode_counter+=1
        
        # Run env with initial config for more significant tensorboard logging 
        if self.det_random_test:
            self.randomize_state = self.config['environment']['randomize_state_init']
            self.do_random_test = False
            if self.episode_counter % self.random_test_intervall == 0:
                self.randomize_state = False
                self.do_random_test = True

                self.start_stop = self.config['environment']['start']
                self.end_stop = self.config['environment']['stop']
                self.targets = self.config['environment']['targets']
                self.reverse_targets = self.targets[::-1][1:] + [0]
                self.schedule = self.config['environment']['schedule']
                self.reverse_schedule = self.schedule[::-1]
                self.create_infra()

        self.time = 0  # time in s
        speeds = [0]*self.n_vehicles
        idxs = [0]*self.n_vehicles      

        if self.randomize_state and not self.baseline:
            self.create_random_track_section()
            self.create_infra()
            self.positions = [0 if x % 2 == 0 else max(self.targets) for x in range(self.n_vehicles)]

            if self.random_pos:
                for x in range(self.n_vehicles):
                    self.positions[x], speeds[x], idxs[x] = self.create_random_pos_speed((x%2!=0))
            
            self.max_step_reward = self.calc_max_step_reward()
        
        else:
            self.positions = [0 if x % 2 == 0 else max(self.targets) for x in range(self.n_vehicles)]

        self.section_list = [0]+self.targets
        self.reverse_section_list = [max(self.targets)] + self.reverse_targets


        next_dist_traveled_breaks = [0] * self.n_vehicles
        next_dist_traveled_cruises = [0] * self.n_vehicles
        next_dist_traveled_accs = [0] * self.n_vehicles

        for x in range(self.n_vehicles):
            next_dist_traveled_breaks[x] = max(0, 0.5 * self.max_break * self.time_steps ** 2 + self.time_steps * speeds[x])
            next_dist_traveled_cruises[x] = max(0, self.time_steps * speeds[x])
            next_dist_traveled_accs[x] = max(0, 0.5 * self.max_acc * self.time_steps ** 2 + self.time_steps * speeds[x])

        self.state = np.array([])
        for x in range(self.n_vehicles):
            if x % 2 == 0:
               self.state = np.concatenate((self.state,np.array([
                                    speeds[x],                                                                                                                  # speed 
                                    self.lower_limit[self.positions[x]],                                                                                        # Lower speed limit
                                    self.speed_limits[self.positions[x]],                                                                                       # current speed limit
                                    self.speed_limits[self.positions[x]] - speeds[x],                                                                           # Difference to current speed limit
                                    self.get_next_speed_limit(self.positions[x], reverse=False),                                                                # next speed limit 
                                    self.get_next_speed_limit(self.positions[x], reverse=False) - speeds[x],                                                    # Difference to next speed limit
                                    min((abs(y - self.positions[x]) for y in self.speed_limits_positions if y > self.positions[x]), default=0),                 # distance to next speed limit
                                    max(0, self.time_steps * speeds[x]),                                                                                        # Distance traveled in current timestep
                                    next_dist_traveled_breaks[x],                                                                                               # next step distance traveled with acc -1
                                    next_dist_traveled_cruises[x],                                                                                              # next step distance traveled with acc 0
                                    next_dist_traveled_accs[x],                                                                                                 # next step distance traveled with acc 1
                                    min((abs(target - self.positions[x]) for target in self.targets if target > self.positions[x]), default=0),                 # distance to next target
                                    self.schedule[idxs[x]],                                                                                                     # remaining time to next target
                                    self.gradient_perc[self.positions[x]] if self.config['rewards']['energy']['step']['power']['use_track_gradient'] else 0,    # Current elevation
                                    self.calc_power(speeds[x],
                                            0,
                                            self.gradient_perc[self.positions[x]] if self.config['rewards']['energy']['step']['power']['use_track_gradient'] else 0),  # current power consumption
                                    self.calc_power(speeds[x]+self.max_break,
                                            self.max_break,
                                            self.gradient_perc[min(int(round(self.positions[x]+next_dist_traveled_breaks[x])), len(self.gradient_perc) - 1)] if self.config['rewards']['energy']['step']['power']['use_track_gradient'] else 0), # next step power consumption with acc -1
                                    self.calc_power(speeds[x],
                                            0,
                                            self.gradient_perc[min(int(round(self.positions[x]+next_dist_traveled_cruises[x])), len(self.gradient_perc) - 1)] if self.config['rewards']['energy']['step']['power']['use_track_gradient'] else 0),# next step power consumption with acc 0
                                    self.calc_power(speeds[x]+self.max_acc,
                                            self.max_acc,
                                            self.gradient_perc[min(int(round(self.positions[x]+next_dist_traveled_accs[x])), len(self.gradient_perc) - 1)] if self.config['rewards']['energy']['step']['power']['use_track_gradient'] else 0)# next step power consumption with acc 1
                                    ])))
            else:
                self.state = np.concatenate((self.state,np.array([
                                    speeds[x],                                                                                                                  # speed 
                                    self.lower_limit[self.positions[x]],                                                                                        # Lower speed limit
                                    self.reverse_speed_limits[self.positions[x]],                                                                               # current speed limit
                                    self.reverse_speed_limits[self.positions[x]] - speeds[x],                                                                   # Difference to current speed limit
                                    self.get_next_speed_limit(self.positions[x], reverse=True),                                                                 # next speed limit                                               
                                    self.get_next_speed_limit(self.positions[x], reverse=True) - speeds[x],                                                     # Difference to next speed limit
                                    min((abs(y - self.positions[x]) for y in self.reverse_speed_limits_positions if y < self.positions[x]), default=0),         # distance to next speed limit
                                    max(0, self.time_steps * speeds[x]),                                                                                        # Distance traveled in current timestep
                                    next_dist_traveled_breaks[x],                                                                                               # next step distance traveled with acc -1
                                    next_dist_traveled_cruises[x],                                                                                              # next step distance traveled with acc 0
                                    next_dist_traveled_accs[x],                                                                                                 # next step distance traveled with acc 1
                                    min((abs(target - self.positions[x]) for target in self.reverse_targets if target < self.positions[x]), default=0),         # distance to next target
                                    self.reverse_schedule[idxs[x]],                                                                                             # remaining time to next target
                                    -self.gradient_perc[self.positions[x]] if self.config['rewards']['energy']['step']['power']['use_track_gradient'] else 0,   # Current elevation
                                    self.calc_power(speeds[x],
                                            0,
                                            -self.gradient_perc[self.positions[x]] if self.config['rewards']['energy']['step']['power']['use_track_gradient'] else 0),  # current power consumption
                                    self.calc_power(speeds[x]+self.max_break,
                                            self.max_break,
                                            -self.gradient_perc[max(int(round(self.positions[x]-next_dist_traveled_breaks[x])), 0)] if self.config['rewards']['energy']['step']['power']['use_track_gradient'] else 0), # next step power consumption with acc -1
                                    self.calc_power(speeds[x],
                                            0,
                                            -self.gradient_perc[max(int(round(self.positions[x]-next_dist_traveled_cruises[x])), 0)] if self.config['rewards']['energy']['step']['power']['use_track_gradient'] else 0), # next step power consumption with acc 0
                                    self.calc_power(speeds[x]+self.max_acc,
                                            self.max_acc,
                                            -self.gradient_perc[max(int(round(self.positions[x]-next_dist_traveled_accs[x])), 0)] if self.config['rewards']['energy']['step']['power']['use_track_gradient'] else 0) # next step power consumption with acc 1
                                    ])))

        self.state = np.append(self.state, 0)

        self.rewards_achieved_list = [[False] * len(self.targets) for _ in range(self.n_vehicles)]

        # Copy the schedule for each tram in forward direction
        self.remaining_time_per_tram = [self.schedule.copy() for _ in range(self.n_vehicles)]

        # Copy the schedule for each tram in reverse direction
        self.reverse_remaining_time_per_tram = [self.reverse_schedule.copy() for _ in range(self.n_vehicles)]
        
        self.cumsum_energy = 0

        self.pos_return = 0
        self.vel_return = 0
        self.sched_return = 0
        self.power_return = 0

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

    def create_random_pos_speed(self, rev = False):
        position = random.randint(0,max(self.targets))
        max_break = abs(self.max_break)
        sched_factor = self.config['environment']['schedule_factor']

        if rev:
            idx, closest_target = [(index, target) for index, target in enumerate(self.reverse_targets) if target <= position][0]
            dist_target = abs(closest_target - position)
            self.reverse_schedule[idx] = int(dist_target*sched_factor) + random.randint(0,40)
            current_speed_limit = int(self.reverse_speed_limits[max(int(round(position)), 0)])
        else:
            idx, closest_target = [(index, target) for index, target in enumerate(self.targets) if target >= position][0]
            dist_target = abs(closest_target - position)
            self.schedule[idx] = int(dist_target*sched_factor) + random.randint(0,40)
            current_speed_limit = int(self.speed_limits[min(int(round(position)), len(self.speed_limits) - 1)])
        
        for x in range(current_speed_limit+1):
            stopping_distance = (x ** 2) / (2 * max_break)
            if stopping_distance > dist_target:
                x-=1
                break

        return position, random.randint(0,x), idx
