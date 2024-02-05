# Tram_RL

OpenAI gym based RL-Environment to train PPO agents to learn how to energy efficiently control multiple tramways. This repository includes the follwoing:
- MultiTram environment with multi-discrete action space MultiDiscrete([3]*n_trams)
- Modified cleanRL PPO agent to account for multi-discrete action space.
- Experiment Runner to run multiple PPO training runs in batches.
- Hyperparameter tuning tool.

## Setup
- Create virtual python env (conda create --name Tram_RL python=3.10.13 )
- Python Version: 3.10.13
- Install requirements.txt (pip install -r requirements.txt)

- (OPTIONAL) To install cuda either use pip or conda
    - pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
    - conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge

## Single runs
- Modify config.yaml in "/src/agents" according to requirements
- cd to "/src/agents"
- Run "python ppo_MT.py --help" for args
- To retrain an agent use "--retrain" and store agent as "agent1.pt" in "src/agents/utils/agents/MultiTram"
- Results are stored in "/src/agents/runs/ppo/"

## Batch Runs with Experiment_Runner v1.1
- Run multiple experiments for defined value ranges
- Modify ER_config.yaml in /src according to requirements
- cd to "/src"
- Run "python experiment_runner.py --help" for available commands
- Use "--max-runs (int)" to determine the number of runs to be started in parallel. If number of value combination > max-runs, runs will be run in batches
- Results are stored in "/src/runs/ppo/"

- Specify value ranges in src/ER_config.yaml
- Ways to specify value range:
    - Just one value: List of len 1 in yaml
    ```
    rewards:
      positional:
        calculation: "1/(speed+1) * value"
        value_range: 
          - 100
        value_steps: 1
    ```
    ```
    Output: Range of Values: [100]
    ```
    - Range of values: List of len 2 in yaml (value_range) + total values (value_steps) of range. Values in value_range are first and last value. Diff to value steps number of values are interpolated between value_range
    ```
    rewards:
      positional:
        calculation: "1/(speed+1) * value"
        value_range: 
          - 100
          - 1000         
        value_steps: 3
    ```
    ```
    Output: Range of Values: [100.0, 550.0, 1000.0]
    ```
    - Specific range of values: List of len n in yaml (value_range) + value steps = len n 
    ```
    rewards:
      positional:
        calculation: "1/(speed+1) * value"
        value_range: 
          - 100
          - 890           
          - 1000         
        value_steps: 3
    ```
    ```
    Output: Range of Values: [100.0, 890.0, 1000.0]
    ```

## HP Search v1.1
- Run hyperparameter tuning.
- Modify HP_config.yaml in /src according to requirements
- Cd to "/src"
- Run "python hp_search.py --help" for available commands
- Use "--num-exp INT" to specify number of tuning runs
- To retrain an agent specify in HP_config.yaml and store agent as "agent1.pt" in src/agents/utils/agents/MultiTram
- Results are stored in "/src/runs/ppo/"
