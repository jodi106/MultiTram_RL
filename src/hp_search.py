import subprocess
import sys
import os
import time
import yaml
import json
import argparse
from argparse import RawTextHelpFormatter
from distutils.util import strtobool
from itertools import product
import random

random.seed(42)

from concurrent.futures import ThreadPoolExecutor

HP_RUNNER_VERSION = '1.1'

class ExperimentRunner(object):
    def __init__(self, args:dict, max_parallel_runs, env_id, exp_name, seed, num_exp, multi_tram):
        
        self.config = self.get_config()
        self.args = args
        self.max_parallel_runs = max_parallel_runs

        self.env_id = env_id
        self.exp_name = exp_name
        self.seed = seed

        self.num_exp = num_exp
        self.multi_tram = multi_tram

        self.store_config = True

    def get_config(self):
        with open(os.path.dirname(os.path.abspath(__file__))+ "\\HP_config.yaml", "r") as yamlfile:
            config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        return config


    def build_hp_dict(self, num_exp):
        hp_dict = {
            "learning-rate": [2.5e-4, 0.0003, 0.0001],
            "num-envs": [24, 4, 64, 1],
            "anneal-lr": [True, False],
            "gamma": [0.995, 1, 0.99, 0.95], # initial 0.995
            "gae-lambda": [0.95, 0.9],
            "num-minibatches": [4,2,8],
            "update-epochs": [4, 2, 8],
            "norm-adv": [True, False],
            "clip-coef": [0.2, 0.25, 0.3, 0.1, 0.5],
            "clip-vloss": [True, False],
            "ent-coef": [0],#0.01, 0.02, 0.005, 0.0001], # initial 0.01
            "vf-coef": [0.5, 0.6, 0.4, 0.8, 0.3],
            "max-grad-norm": [0.5, 0.6, 0.4],
            #"units": [64, 32, 128, 256, 320]
            #"target-kl": [None]
                    }
        
        # HP Search deterministic and random search space.
        combo_dicts = self.generate_param_combinations(hp_dict)
        combo_dicts += self.generate_random_param_combinations(hp_dict, num_exp-len(combo_dicts))
        
        # HP Search only random search space
        #combo_dicts = self.generate_random_param_combinations(hp_dict, num_exp)

        return combo_dicts

    def generate_param_combinations(self, hp_dict, specific_key=None):
        param_dicts = []
        # Generate all possible combinations for each key
        for key, values in hp_dict.items():
            if len(values) > 1:
                combinations_for_key = list(product(values[1:]))
            else:
                continue
            
            # Create dictionaries with the current key and all first parameters of other keys
            for combination in combinations_for_key:
                if specific_key is None or specific_key == key:
                    param_dict = {k: v[0] if k != key else combination[0] for k, v in hp_dict.items()}
                    param_dicts.append(param_dict)

        return param_dicts

    def generate_random_param_combinations(self, hp_dict, n):
        param_dicts = []

        for _ in range(n):
            param_dict = {key: random.choice(values) for key, values in hp_dict.items()}
            param_dicts.append(param_dict)

        return param_dicts

    def build_new_config(self, combo_dict: dict, run_nr:int):
        run_name = f"{self.env_id}__{self.exp_name}__{self.seed}__{run_nr}"

        if not os.path.exists(f"runs/ppo/{self.exp_name}/{run_name}"):
            os.makedirs(f"runs/ppo/{self.exp_name}/{run_name}")


        with open(f"runs/ppo/{self.exp_name}/{run_name}/config.yaml", 'w') as yaml_file:
            yaml.dump(self.config, yaml_file, default_flow_style=False, sort_keys=False)

        with open(f"runs/ppo/{self.exp_name}/{run_name}/parameters.json", "w") as json_file:
            json.dump(combo_dict, json_file, indent=4, separators=(',', ': '))
        
        return run_name

    def build_cmd(self, run_name, combo_dict):
        num_steps = self.config['environment']['num_steps']
        max_steps = self.config['environment']['max_steps']

        retrain = self.config['environment']['retrain_agent']

        version = "ppo"
        if self.multi_tram:
            version = "ppo_MT"

        cmd = f'python agents/{version}.py --total-timesteps {num_steps} --num-steps {max_steps} --run-name {run_name} --ER --retrain {retrain} '
        for arg in self.args.keys():
            if "max-runs" not in arg and "num-exp" not in arg and "multi-tram" not in arg:
                cmd += f'--{arg} {self.args[arg]} '
        for arg in combo_dict.keys():
            cmd += f'--{arg} {combo_dict[arg]} '
    
        return cmd
    
    def run(self):
        combinations = self.build_hp_dict(self.num_exp)
        print(f"[INFO] Starting {len(combinations)} runs")

        with ThreadPoolExecutor(max_workers=self.max_parallel_runs) as executor:
            futures = []

            for n, combo in enumerate(combinations):
                path = self.build_new_config(combo, n)
                cmd = self.build_cmd(path, combo)
                print(f"[INFO] Adding run [{n}|{len(combinations)}] to ThreadPool: {combo}")
                futures.append(executor.submit(subprocess.run, cmd, shell=True))
                #time.sleep(1)

            for future in futures:
                future.result()

def main():
    description = ("Hyperparameter Runner for TramRL: Run and Evaluate multiple experiments on TramRL env based on defined value ranges\n"
                   "Current version: " + HP_RUNNER_VERSION)
   
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=RawTextHelpFormatter)
    # Experiment Runner specific arguments
    parser.add_argument('-v', '--version', action='version', version='%(prog)s ' + HP_RUNNER_VERSION)
    parser.add_argument("--max-runs", type=int, default=10,
        help="the number of parallel runs started by Experiment runnner. If number of value combination > max-runs, runs will be run in batches")
    parser.add_argument("--num-exp", type=int, default=100,
        help="the number of total experiments to run. If number of parameters < num-exp, difference will be filled with random combinations of parameters ")
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

    args = parser.parse_args()
    non_default_values = {arg.replace("_", "-"): value for arg, value in vars(args).items() if value != parser.get_default(arg)}

    runner = ExperimentRunner(non_default_values, args.max_runs, args.env_id, args.exp_name, args.seed, args.num_exp, args.multi_tram)
    runner.run()

if __name__ == "__main__":
    sys.exit(main())