import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import yaml
import json
#from agents.environments.TramEnv_n_stop import TramEnv
from agents.environments.Multi_TramEnv_n_stop import TramEnv as MultiTramEnv
from agents.utils.baseline_agent import DeterministicAgent


class Plotter():
    def __init__(self, directory, multi_tram=True, n_vehicles = 1):
        self.directory = directory
        self.multi_tram = multi_tram
        self.n_vehicles = n_vehicles
        with open(self.directory +'/config.yaml', "r") as yamlfile:
            self.config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        if multi_tram:
            self.env = MultiTramEnv(self.config, baseline=True)
        else:
            raise Exception("Not Implemented")
            self.env = TramEnv(self.config, baseline=True)

    def create_plot(self):
        if self.multi_tram:
            agent_data = self.get_mt_data()
        else:
            agent_data = self.get_st_data()
        
        sb_data = self.create_baseline_vals()
        ob_data = self.create_optimized_baseline_vals()

        plt.figure(figsize=(10, 5))
        plt.plot(self.env.lower_limit, color="lightgrey",  label="Lower Speed Limit")
        plt.plot(self.env.speed_limits, color="darkgrey",label="Upper Speed Limit")

        # Baseline
        for idx, (key, value) in enumerate(sb_data.items()):
            if idx == 1:  # Start at the second key
                plt.plot(sb_data["Position"], value, color='blue', linestyle='--', label=f"SB {key}")   
        #print(sb_data["remain_time"])
        # Optimized Baseline
        for idx, (key, value) in enumerate(ob_data.items()):
            if idx == 1:  # Start at the second key
                plt.plot(ob_data["Position"], value, color='red', linestyle='--', label=f"OB {key}")   
        #print(ob_data["remain_time"])
        # Agent
        colors = ["limegreen", "green", 'olive', 'darkolivegreen']
        for x in range(self.n_vehicles):
            for idx, (key, value) in enumerate(agent_data.items()):
                if idx == 1:  # Start at the second key
                    plt.plot(agent_data["Position"][f"{x}"], value[f"{x}"], color=colors[x], linestyle='-', label=f"PPO {key}_{x}")

        #print(agent_data["RemainingTime/10"]["1"])
        print(agent_data['Position']["1"])
        print(agent_data['Speed']["1"])

        # Adding legends, titles, and labels
        plt.legend(loc="upper left")
        plt.xlabel("Position in [m]")
        plt.ylabel("Speed in [m/s]")
        plt.tight_layout()
        plt.show()

        # Define offsets for text annotations to prevent overlap
        text_offset_x = 100  # Horizontal offset
        text_offset_y = 0  # Vertical offset

        # Plotting energy consumption
        plt.figure(figsize=(10, 5))

        # Agent
        colors = ["limegreen", "green", 'darkgreen']
        for x in range(1):
            plt.plot(agent_data["Position"][f"{x}"], agent_data["energy_consumption"], color=colors[x], linestyle='-', label="PPO Consumption")
            # Add 'x' marker and text for last value with offset
            plt.scatter(agent_data["Position"][f"{x}"][-1], agent_data["energy_consumption"][-1], color='limegreen', marker='x')
            plt.text(agent_data["Position"][f"{x}"][-1] - text_offset_x, agent_data["energy_consumption"][-1] - 0, 
                    f'{agent_data["energy_consumption"][-1]:.2f} kWh', color='limegreen', verticalalignment='bottom')

        # Baseline
        plt.plot(sb_data["Position"], sb_data["cum_energy"], color="blue", linestyle='--', label="SB Consumption")
        # Add 'x' marker and text for last value with offset
        plt.scatter(sb_data["Position"][-1], sb_data["cum_energy"][-1], color='blue', marker='x')
        plt.text(sb_data["Position"][-1] - text_offset_x, sb_data["cum_energy"][-1] - text_offset_y, 
                f'{sb_data["cum_energy"][-1]:.2f} kWh', color='blue', verticalalignment='bottom')

        # Optimized Baseline
        plt.plot(ob_data["Position"], ob_data["cum_energy"], color="red", linestyle='--', label="OB Consumption")
        # Add 'x' marker and text for last value with offset
        plt.scatter(ob_data["Position"][-1], ob_data["cum_energy"][-1], color='red', marker='x')
        plt.text(ob_data["Position"][-1] - text_offset_x, ob_data["cum_energy"][-1] - 2.5 , 
                f'{ob_data["cum_energy"][-1]:.2f} kWh', color='red', verticalalignment='bottom')

        plt.legend(loc="upper left")
        plt.ylabel("Cumulated Energy Consumption in [kWh]")
        plt.xlabel("Position in [m]")
        plt.tight_layout()
        plt.show()

        plt.close()

    def get_st_data(self):
        raise Exception("Not Implemented")
        results = pd.read_csv(self.directory + '/results.csv') 
        data = {
        'Position': np.array(results['Position']),
        'SpeedProfil': np.array(results['SpeedProfil']),
        'RemainingTime/10': np.array(results['RemainingTime/10']),
        'energy_consumption': np.array(results['energy_consumption']),
        #'SpeedReward': df['SpeedReward'],
        #'PosReward': df['PosReward'],
        #'ScheduleReward': df['ScheduleReward'],
        #'EnergyReward': df['EnergyReward']
        }
        return data

    def get_mt_data(self):
        with open(self.directory + "/results.json") as f:
            d = json.load(f)

        data = {
            "Position": d["Positions"],
            "Speed": d["Speeds"],
            "RemainingTime/10": d["Times"],
            'energy_consumption': d["Cum_Total_Energy"]
        }

        return data

    def create_baseline_vals(self):
        if self.multi_tram:
            position,speed,cum_energy, time = self.run_baseline_MT()[:4]
        else:
            position,speed,cum_energy, time = self.run_baseline()[:4]

        data = {
            "Position": position,
            "Speed": speed,
            "cum_energy": cum_energy,
            "remain_time": time
        }

        return data
    
    def create_optimized_baseline_vals(self):
        index = self.optimize_baseline()
        if self.multi_tram:
            position,speed,cum_energy, time = self.run_baseline_MT(max_speed=index)[:4]
        else:
            position,speed,cum_energy, time = self.run_baseline(max_speed=index)[:4]

        data = {
            "Position": position,
            "Speed": speed,
            "cum_energy": cum_energy,
            "remain_time": time
        }

        return data

    def run_baseline(self, max_speed = None):
        raise Exception("Not Implemented")
        env = TramEnv(self.config, baseline=True)
        deterministic_agent = DeterministicAgent(env, max_speed)
        state = env.reset()
        done = False
        obs_dict = {}
        info_dict = {}
        step_num = 0
        episode_reward = 0

        while not done:
            obs_dict[step_num] = [env.position] + list(env.get_state())
            action = deterministic_agent.act()
            state, reward, done, info = env.step(action)
            info_dict[step_num] = info
            episode_reward += reward
            step_num += 1

        print(f"Dissipated for speed{max_speed}: {sum([obs[15]/3600 for obs in obs_dict.values() if obs[15] < 0])}")

        return  ([obs[0] for obs in obs_dict.values()],
                 [obs[1] for obs in obs_dict.values()],
                [info["cumsum_energy_consumption"] for info in info_dict.values()], 
                [obs[13] for obs in obs_dict.values()],
                episode_reward
                )

    def run_baseline_MT(self, max_speed = None):
        env = MultiTramEnv(self.config, baseline=True)
        deterministic_agent = DeterministicAgent(env, max_speed)
        state = env.reset()
        done = False
        info_dict = {}
        step_num = 0
        episode_reward = 0

        while not done:
            action = deterministic_agent.act_MT()
            state, reward, done, info = env.step(action)
            info_dict[step_num] = info
            episode_reward += reward
            step_num += 1

        return  ([info["State"]["Positions"][0] for info in info_dict.values()],
                [info["State"]["Speed_0"] for info in info_dict.values()],
                [info["cumsum_energy_consumption"] for info in info_dict.values()], 
                [info["State"]["Remain_Time_0"] for info in info_dict.values()],
                episode_reward)        


    def optimize_baseline(self):
        returns = [] # Returns
        if self.multi_tram:
            for x in range(20):
                returns.append(self.run_baseline_MT(max_speed=x)[4])
        else:    
            for x in range(20):
                returns.append(self.run_baseline(max_speed=x)[4])


        return returns.index(max(returns[4:]))

if __name__ == "__main__":
    directory = r"PATH_TO_RUN_DIRECTORY"
    plotter = Plotter(directory,
                      n_vehicles = 2)
    plotter.create_plot()



