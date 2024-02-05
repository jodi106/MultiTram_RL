import os
import yaml
import json
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from torch.distributions.categorical import Categorical
from environments.Multi_TramEnv_n_stop import TramEnv
from utils.baseline_agent import DeterministicAgent

class Evaluator(object):
    def __init__(self, run_name, directory, config_directory, units, multiTram=False):
        self.run_name = run_name
        self.directory = directory
        self.config_directory = config_directory
        self.multi_tram = multiTram
        self.units = units

        self.config = self.get_directory()

    def get_directory(self):
        with open(self.config_directory, "r") as yamlfile:
        #with open(r"C:\Users\Admin\OneDrive - thu.de\Dokumente\00_Masterarbeit\01_Code\Tram_RL\src\agents\config.yaml", "r") as yamlfile:    
            return yaml.load(yamlfile, Loader=yaml.FullLoader)

    def evaluate_run(self, nb_episodes=100):
        # Set randomize to false for testing
        self.config['environment']['randomize_state_init'] = False

        env = TramEnv(self.config)
        print("MAX STEP REWARD: ", env.calc_max_step_reward())
        agent = Eval_Agent(env, self.units)
        agent.load_state_dict(torch.load(f"{self.directory}/agent.pt"))
        #agent.load_state_dict(torch.load(r"C:\Users\Admin\OneDrive - thu.de\Dokumente\00_Masterarbeit\01_Code\Tram_RL\src\agents\runs\ppo\MultiTram_Test\agent.pt"))
        agent.eval()

        # Eval deterministic:
        self.env_testing(env,agent,deterministic=True,nb_episodes=1)
        # Eval stochastic:
        self.env_testing(env,agent,deterministic=False,nb_episodes=nb_episodes)

    def env_testing(self, env, agent, deterministic:bool, nb_episodes):
        rewards = []
        obs_dict = {}
        info_dict = {}

        for episode in range(nb_episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            step_info = {}
            step_num = 0

            while not done:
                obs = torch.FloatTensor(obs).unsqueeze(0)
                
                with torch.no_grad():
                    action, _, _, _ = agent.get_action_and_value(obs, deterministic)

                obs, reward, done, info = env.step(action.cpu().numpy().flatten())
                step_info[step_num] = info
                episode_reward += reward
                step_num += 1

            print(f"Episode {episode + 1}: Total Reward: {episode_reward}")
            rewards.append(episode_reward)
            info_dict[episode] = step_info
            end_time = time.time() 

        self.plot_results(env, rewards, info_dict, deterministic)
        if deterministic:
            best_baseline_speed = self.create_baseline_overview(env)
            self.run_baseline(env)
            self.run_baseline(env, max_speed=best_baseline_speed, slow=True)
        
    def plot_results(self, env, rewards, info_dict, deterministic, baseline=False, slow=False):
        if baseline:
            best_episode_info = info_dict
        else:
            best_episode_idx = np.argmax(rewards)
            best_episode_info = info_dict.get(best_episode_idx, {})

        data = {"Positions": {},
                "Speeds": {},
                "Powers": {},
                "Times": {},
                "Consumption": {},
                "Cum_Total_Energy": {},
                "total_reward": {},
                "speed_reward": {},
                "pos_reward": {},
                "schedule_reward": {},
                "energy_reward": {},
                "recuperated_power": {},
                "Diss_Power": {}}
        
        for x in range(env.n_vehicles):
            data["Positions"][x] = [float(obs["State"]["Positions"][x]) for obs in best_episode_info.values()]
            data["Speeds"][x] = [float(obs["State"][f"Speed_{x}"]) for obs in best_episode_info.values()]
            data["Times"][x] = [float(obs["State"][f"Remain_Time_{x}"] / 10) for obs in best_episode_info.values()]
            data["Powers"][x] = [float(obs["State"][f"Power_{x}"]) for obs in best_episode_info.values()]
            data["Consumption"][x] = [float(y / 3600) if y > 0 else 0.0 for y in data["Powers"][x]]

        data["Cum_Total_Energy"] = [float(obs["cumsum_energy_consumption"]) for obs in best_episode_info.values()]
        data["total_reward"] = [float(obs["total_reward"]) for obs in best_episode_info.values()]
        data["speed_reward"] = [float(obs["speed_reward"]) for obs in best_episode_info.values()]
        data["pos_reward"] = [float(obs["pos_reward"]) for obs in best_episode_info.values()]
        data["schedule_reward"] = [float(obs["schedule_reward"]) for obs in best_episode_info.values()]
        data["energy_reward"] = [float(obs["energy_reward"]) for obs in best_episode_info.values()]
        data["recuperated_power"] = [float(obs["recuperated_power"]) for obs in best_episode_info.values()]
        data["Diss_Power"] = [float(obs["dissipated_power"]) for obs in best_episode_info.values()]

        addon = "_d" if deterministic else ""

        if not baseline:
            with open(f"{self.directory}/results{addon}.json", "w") as json_file:
                json.dump(data, json_file)

        # Step Rewards Plots
        fig, ax1 = plt.subplots(figsize=(16, 9))

        ax1.plot(data["speed_reward"], label=f"speed_reward")
        ax1.plot([x/1000 for x in data["pos_reward"]], label=f"pos_reward")
        ax1.plot(data["schedule_reward"], label=f"schedule_reward_")
        ax1.plot([x/1000 for x in data["energy_reward"]], label=f"energy_reward")
        ax1.plot([x/1000 for x in data["recuperated_power"]], label=f"recuperated_power")
        ax1.plot([x/1000 for x in data["Diss_Power"]], label=f"Diss_Power [MW]")

        title_addon = "Baseline" if baseline else ("Deterministic Eval" if deterministic else "Stochastic Eval")
        addon = "_baseline" if baseline else ("_d" if deterministic else "_s")

        if slow:
            title_addon = "Baseline Slow"
            addon = "_baselineSlow"

        ax1.set_title(f'PPO Rewards {title_addon} Max Return: {max(rewards)}, Steps: {len(data["Diss_Power"])}')
        ax1.set_xlabel('Steps in [s]')
        ax1.set_ylim(-10, 20)

        # Set the first y-axis label and legend
        ax1.set_ylabel('Reward')
        ax1.legend(loc="upper left")

        # Create a second y-axis on the right side
        ax2 = ax1.twinx()
        ax2.set_ylabel('Speed [m/s]')

        for x in range(env.n_vehicles):
            ax2.plot(data["Speeds"][x], label=f"Speeds_{x}", linestyle='dotted')

        # You can adjust the limits for the second y-axis as needed
        ax2.set_ylim(-10, 20)
        ax2.legend(loc="upper right")
        #plt.show()
        plt.savefig(f'{self.directory}/eval_steps{addon}.png')
        plt.close()

        # Eval Plots
        plt.figure(figsize=(20,12))
        for x in range(env.n_vehicles):
            plt.plot(data['Positions'][x], data["Speeds"][x], label=f"Speeds_{x}")
            plt.plot(data['Positions'][x], data["Times"][x], label=f"Remaining Time/10_{x}")
        plt.plot(env.lower_limit, label = "Lower Speed Limit")
        plt.plot(env.speed_limits, label = "Upper Speed limit")

        plt.title(f'PPO {title_addon} Max Return: {max(rewards)}, Steps: {len(data["Diss_Power"])}')
        plt.xlabel('Position [m]')
        plt.ylabel('Speed [m/s] | Reward')
        plt.legend()
        plt.ylim(-15, 25)
        #plt.show()
        plt.savefig(f'{self.directory}/eval{addon}.png')
        plt.close()

        # Energy Consumption Plot
        plt.figure(figsize=(20,12))

        for x in range(env.n_vehicles):
            plt.plot(data['Positions'][x],np.array(data['Consumption'][x]).cumsum(), label=f"Cumsum Consumption_{x}")
        plt.plot(data['Positions'][x],[x/3600 for x in data["recuperated_power"]], label=f"Recuperated Energy")   

        plt.title(f'''PPO {title_addon} Energy Consumption {data["Cum_Total_Energy"][-1]}, 
                  Recuperated Energy: {sum([x/3600 for x in data["recuperated_power"]])},
                  Dissipated Energy: {sum([x/3600 for x in data["Diss_Power"]])}''')
        plt.xlabel('Position [m]')
        plt.ylabel('Energy Consumption [kWh]')
        plt.legend()
        plt.savefig(f'{self.directory}/consumption_{addon}.png')
        plt.close()

        if not baseline:
            self.export_result_dict(data, addon, rewards)

    def export_result_dict(self, data, addon, rewards):
            # Write results to json file
            result_dict = {
                "Run": self.run_name+addon,
                "Return": max(rewards),
                "Steps": len(data["Positions"][0]),
                "Energy_Consumption": data["Cum_Total_Energy"][-1],
                "Config":self.config}

            try:
                with open(f'{os.path.dirname(self.directory)}/runs_overview_{addon}.json', 'a') as json_file:
                    json.dump(result_dict, json_file)
                    json_file.write('\n')
            except Exception as e:
                print(f"[WARNING] Could not save results.json due to error: {e}")

    def run_baseline(self, env, max_speed = None, slow=False, plot=True):
        env = TramEnv(self.config, baseline=True)
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

        if plot:
            self.plot_results(env, [0,episode_reward], info_dict, deterministic=True, baseline=True, slow=slow)

        else:
            return  ([info["State"]["Positions"] for info in info_dict.values()],
                    [info["cumsum_energy_consumption"] for info in info_dict.values()], 
                    [info["energy_reward"] for info in info_dict.values()], 
                    episode_reward)        

    def create_baseline_overview(self, env):
        vals = [] # Returns
        vals2 = [] # Energy Consumptions
        for x in range(20):
            retuns_vals = self.run_baseline(env=env,max_speed=x,plot=False) 
            vals.append(retuns_vals[3])
            vals2.append(retuns_vals[1][-1])

        fig, ax1 = plt.subplots(figsize=(10,5))

        ax1.plot(vals, label="Episode Return", color="blue")
        ax1.scatter(vals.index(max(vals[4:])), max(vals[4:]), color="red", label=f"Return Speed {vals.index(max(vals[4:]))}: {round(max(vals[4:]), 4)} ")
        ax1.set_xlabel("Max Speed Baseline [m/s]")
        ax1.set_ylabel("Episode Return", color="blue")
        ax1.tick_params(axis='y', labelcolor="blue")
        ax1.set_title("Episode Return and Consumption for different Baseline Speeds")

        ax2 = ax1.twinx()
        ax2.plot(vals2, label="Cumsum Energy consumption", color="green")
        ax2.scatter(vals2.index(min(vals2[4:])), min(vals2[4:]), color="orange", label=f"Consumption Speed {vals2.index(min(vals2[4:]))}: {round(min(vals2[4:]), 4)} kWh")
        ax2.set_ylabel("Cumsum Energy consumption [kWh]", color="green")
        ax2.tick_params(axis='y', labelcolor="green")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        lines = lines1 + lines2
        labels = labels1 + labels2
        ax1.legend(lines, labels, loc="lower right")

        plt.savefig(f'{self.directory}/Baseline_Overview.png')
        plt.close()

        return vals.index(max(vals[4:]))


class Eval_Agent(nn.Module):
    def __init__(self, envs, units):
        super().__init__()
        self.nvec = envs.action_space.nvec
        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(units, units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(units, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            self.layer_init(nn.Linear(np.array(envs.observation_space.shape).prod(), units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(units, units)),
            nn.Tanh(),
            self.layer_init(nn.Linear(units, self.nvec.sum()), std=0.01),
        )
        

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, deterministic, action=None):
        logits = self.actor(x)
        split_logits = torch.split(logits, self.nvec.tolist(), dim=1)
        multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
        if action is None:
            if deterministic:
                action = torch.stack([torch.argmax(logits, dim=-1) for logits in split_logits]) 
            else:
                action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return action.T, logprob.sum(0), entropy.sum(0), self.critic(x)
    
    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
