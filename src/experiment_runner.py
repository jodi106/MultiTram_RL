import subprocess
import sys
import os
import time
import yaml
import argparse
from argparse import RawTextHelpFormatter
from distutils.util import strtobool
from itertools import product

from concurrent.futures import ThreadPoolExecutor

EXPERIENCE_RUNNER_VERSION = '1.1'

class ExperimentRunner(object):
    def __init__(self, args:dict, max_parallel_runs, env_id, exp_name, seed, multi_tram):
        
        self.config = self.get_config()
        self.experiment_dict = self.get_value_ranges()
        self.args = args
        self.max_parallel_runs = max_parallel_runs

        self.env_id = env_id
        self.exp_name = exp_name
        self.seed = seed
        self.multi_tram = multi_tram

    def get_config(self):
        with open(os.path.dirname(os.path.abspath(__file__))+ "\\ER_config.yaml", "r") as yamlfile:
            config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        return config

    def get_value_ranges(self):
        experiment_dict = {}

        # Positional Rewards
        pos_conf = self.config['rewards']['positional']
        experiment_dict["pos_vals"] = self.interpolate_values(pos_conf['value_range'],
                                                              pos_conf['value_steps'])

        # Speed Rewards
        speed_conf = self.config['rewards']['velocity']
        for key in speed_conf.keys():
            experiment_dict[f"{key}_vals"] = self.interpolate_values(speed_conf[key]['value_range'],
                                                                     speed_conf[key]['value_steps'])
        # Schedule Rewards
        schedule_conf = self.config['rewards']['schedule']
        if schedule_conf['use_sparse']:
            experiment_dict['schedule_vals_sparse'] = self.interpolate_values(schedule_conf['sparse']['value_range'],
                                                                              schedule_conf['sparse']['value_steps'])
        if schedule_conf['use_step']:
            experiment_dict['schedule_vals_step'] = self.interpolate_values(schedule_conf['step']['value_range'],
                                                                            schedule_conf['step']['value_steps'])
        # Energy Rewards
        energy_conf = self.config['rewards']['energy']
        if energy_conf['use_sparse']:
            experiment_dict['energy_vals_sparse'] = self.interpolate_values(energy_conf['sparse']['value_range'],
                                                                            energy_conf['sparse']['value_steps'])
        if energy_conf['use_step']:
            experiment_dict['energy_vals_step'] = self.interpolate_values(energy_conf['step']['power']['value_range'],
                                                                          energy_conf['step']['power']['value_steps'])
        experiment_dict['recup_factor'] = self.interpolate_values(energy_conf['recup_factor']['value_range'],
                                                                energy_conf['recup_factor']['value_steps'])    
        # Stop Zone
        if pos_conf['use_stopZone']:
            experiment_dict['stopZone_vals'] = self.interpolate_values(pos_conf['stop_zone']['value_range'],
                                                              pos_conf['stop_zone']['value_steps'])       
        return experiment_dict

    def interpolate_values(self, value_range, num_vals):
        if len(value_range) == 1 or len(value_range) == num_vals:
            return value_range
        else:
            first_value, second_value = value_range[:2]
            interpolated_values = [first_value + i * (second_value - first_value) / (num_vals - 1) for i in range(num_vals)]
            return interpolated_values

    def build_new_config(self, combo: tuple, run_nr:int):
        # tuple(Pos, speed_reward, speeding_cost, slow_cost, schedule_sparse, schedule_step, energy_step, stop_zone)
        new_config = self.config.copy()
        new_config['rewards']['positional']['value'] = combo[0]
        new_config['rewards']['velocity']['speed_reward'] = combo[1]
        new_config['rewards']['velocity']['speeding_cost'] = combo[2]
        new_config['rewards']['velocity']['slow_cost'] = combo[3]
        index = 4
        if self.config['rewards']['schedule']['use_sparse']:
            new_config['rewards']['schedule']['sparse']['value'] = combo[index]
            index+=1
        if self.config['rewards']['schedule']['use_step']:
            new_config['rewards']['schedule']['step']['value'] = combo[index]
            index+=1
        new_config['rewards']['energy']['step']['power']['value'] = combo[index]
        index +=1
        new_config['rewards']['energy']['recup_factor'] = combo[index]
        index +=1
        if self.config['rewards']['positional']['use_stopZone']:
            new_config['rewards']['positional']['stop_zone']['value'] = combo[index]        

        self.remove_keys(new_config, ['value_range','value_steps'])

        run_name = f"{self.env_id}__{self.exp_name}__{self.seed}__{run_nr}"

        if not os.path.exists(f"runs/ppo/{self.exp_name}/{run_name}"):
            os.makedirs(f"runs/ppo/{self.exp_name}/{run_name}")

        with open(f"runs/ppo/{self.exp_name}/{run_name}/config.yaml", 'w') as yaml_file:
            yaml.dump(new_config, yaml_file, default_flow_style=False, sort_keys=False)
        
        return run_name

    def remove_keys(self, d: dict, keys_to_remove: list):
        if isinstance(d, dict):
            keys = list(d.keys())
            for key in keys:
                if key in keys_to_remove:
                    del d[key]
                else:
                    self.remove_keys(d[key], keys_to_remove)

    def build_cmd(self, run_name):
        num_steps = self.config['environment']['num_steps']
        max_steps = self.config['environment']['max_steps']

        version = "ppo"
        if self.multi_tram:
            version = "ppo_MT"

        cmd = f'python agents/{version}.py --total-timesteps {num_steps} --num-steps {max_steps} --run-name {run_name} --ER '
        for arg in self.args.keys():
            if "max-runs" not in arg and "multi-tram" not in arg:
                cmd += f'--{arg} {self.args[arg]} '

        return cmd

    def run(self):
        combinations = list(product(*self.experiment_dict.values()))
        print(f"[INFO] Starting {len(combinations)} runs")

        with ThreadPoolExecutor(max_workers=self.max_parallel_runs) as executor:
            futures = []

            for n, combo in enumerate(combinations):
                path = self.build_new_config(combo, n)
                cmd = self.build_cmd(path)
                print(f"[INFO] Adding run [{n}|{len(combinations)}] to ThreadPool: {combo}")
                futures.append(executor.submit(subprocess.run, cmd, shell=True))
                #time.sleep(1)

            for future in futures:
                future.result()

def main():
    description = ("Experiment Runner for TramRL: Run and Evaluate multiple experiments on TramRL env based on defined value ranges\n"
                   "Current version: " + EXPERIENCE_RUNNER_VERSION)
   
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawTextHelpFormatter)
    # Experiment Runner specific arguments
    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + EXPERIENCE_RUNNER_VERSION)
    parser.add_argument("--max-runs", type=int, default=10,
        help="the number of parallel runs started by Experiment runnner. If number of value combination > max-runs, runs will be run in batches")
    parser.add_argument("--multi-tram", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle MultiTram Env")
    
    # Agent Settings
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")

    parser.add_argument("--log-interval", type=int, default=1,
        help="Logs Episode info every defined interval and stores it in tensorflow writer")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="TramRL",
        help="the id of the environment")
    #parser.add_argument("--total-timesteps", type=int, default=500000,
    #    help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=4,
        help="the number of parallel game environments")
    #parser.add_argument("--num-steps", type=int, default=500,
    #    help="the number of steps to run in each environment per policy rollout")
    parser.add_argument("--anneal-lr", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggle learning rate annealing for policy and value networks")
    parser.add_argument("--gamma", type=float, default=0.995,
        help="the discount factor gamma")
    parser.add_argument("--gae-lambda", type=float, default=0.95,
        help="the lambda for the general advantage estimation")
    parser.add_argument("--num-minibatches", type=int, default=4,
        help="the number of mini-batches")
    parser.add_argument("--update-epochs", type=int, default=4,
        help="the K epochs to update the policy")
    parser.add_argument("--norm-adv", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles advantages normalization")
    parser.add_argument("--clip-coef", type=float, default=0.2,
        help="the surrogate clipping coefficient")
    parser.add_argument("--clip-vloss", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Toggles whether or not to use a clipped loss for the value function, as per the paper.")
    parser.add_argument("--ent-coef", type=float, default=0.01,
        help="coefficient of the entropy")
    
    parser.add_argument("--anneal-ent", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Anneal entropie from start value.")

    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")

    args = parser.parse_args()
    non_default_values = {arg.replace("_", "-"): value for arg, value in vars(args).items() if value != parser.get_default(arg)}

    runner = ExperimentRunner(non_default_values, args.max_runs, args.env_id, args.exp_name, args.seed, args.multi_tram)
    #test_dict = runner.get_value_ranges()
    runner.run()

if __name__ == "__main__":
    sys.exit(main())