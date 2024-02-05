# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import argparse
import os
import random
import time
import json
from distutils.util import strtobool

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter

import yaml
import matplotlib.pyplot as plt

from environments.Multi_TramEnv_n_stop import TramEnv
from utils.MT_evaluation import Evaluator
from utils.extended_record_episode_statistics import ExtendedRecordEpisodeStatistics

import warnings

# Ignore all DeprecationWarnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning)

# TODO: Add logs

def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default = "MultiTram_Test",#default=os.path.basename(__file__).rstrip(".py"),
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
        help="NOT INPLEMENTED Logs Episode info every defined interval and stores it in tensorflow writer")
    parser.add_argument("--det-test", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="Only log tensorboard with fixed types of episodes (Used for random env init). Disable to log every episode ")

    # NN specific arguments
    parser.add_argument("--units", type=int, default=64,
    help="number of hidden layers in the neural networ") 

    parser.add_argument("--retrain", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Use a pretrained agent to retrain")
    parser.add_argument("--agent-nr", type=int, default=1,
        help="Agent number to use for retraining")  

    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="TramRL",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=2.5e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=8,
        help="the number of parallel game environments")
    parser.add_argument("--num-steps", type=int, default=500,
        help="the number of steps to run in each environment per policy rollout")
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
        help="Anneal entropy from start value.")

    parser.add_argument("--vf-coef", type=float, default=0.5,
        help="coefficient of the value function")
    parser.add_argument("--max-grad-norm", type=float, default=0.5,
        help="the maximum norm for the gradient clipping")
    parser.add_argument("--target-kl", type=float, default=None,
        help="the target KL divergence threshold")
    parser.add_argument("--run-name", type=str, default='results',
        help="Don't modify if directly running PPO")  
    parser.add_argument("--ER", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="Don't use. Will toggle automatically when running Experience Runner.")
    args = parser.parse_args()
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    # fmt: on
    # Additionals

    return args

def make_env(env_id, seed, idx, capture_video, config_path):
    def thunk():
        try:
            with open(config_path, "r") as yamlfile:
            #with open(r"C:\Users\jonas\OneDrive - thu.de\Dokumente\00_Masterarbeit\01_Code\Tram_RL\src\agents\config.yaml", "r") as yamlfile:
                config = yaml.load(yamlfile, Loader=yaml.FullLoader)
        except Exception as e:
            raise Exception("[ERROR] Couldn't load config file: ", e)

        env = TramEnv(config)
        env = ExtendedRecordEpisodeStatistics(env)
        if capture_video:
            print("[WARNING] capture_video is not implemented")
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return thunk

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.nvec = envs.single_action_space.nvec
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), args.units)),
            nn.Tanh(),
            layer_init(nn.Linear(args.units, args.units)),
            nn.Tanh(),
            layer_init(nn.Linear(args.units, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), args.units)),
            nn.Tanh(),
            layer_init(nn.Linear(args.units, args.units)),
            nn.Tanh(),
            layer_init(nn.Linear(args.units, self.nvec.sum()), std=0.01),
        )
        

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        split_logits = torch.split(logits, self.nvec.tolist(), dim=1)
        multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
        if action is None:
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return action.T, logprob.sum(0), entropy.sum(0), self.critic(x)


if __name__ == "__main__":
    args = parse_args()
    run_name = args.run_name
    log_interval = args.log_interval

    config_path = "config.yaml"
    directory = f"runs/ppo/{args.exp_name}"
    if args.ER:
        directory = f"runs/ppo/{args.exp_name}/{run_name}"
        config_path = f"{directory}/config.yaml"

    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )


    writer = SummaryWriter(directory)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, config_path) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.MultiDiscrete), "only MultiDiscrete action space is supported"
    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    if args.retrain:
        #agent.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__)) + f"/utils/agents/MultiTram/agent{args.agent_nr}.pt"))
        agent.load_state_dict(torch.load(os.path.dirname(os.path.abspath(__file__)) + f"/utils/agents/MultiTram/agent{args.agent_nr}.pt", map_location=device))
    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size
    frac_ent = 1

    for update in range(1, num_updates + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow
        
        if args.anneal_ent:
            frac_ent = 1.0 - (update - 1.0) / num_updates

        for step in range(0, args.num_steps):
            global_step += 1 * args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(done).to(device)

            for item in info:
                if "episode" in item.keys():

                    if global_step % 100 == 0:
                        print(f"RUN: {run_name[-4:]}, SPS:", int(global_step / (time.time() - start_time)))
                    print(f"RUN: {run_name[-4:]},global_step={global_step}, episodic_return={item['episode']['r']}")
                    if item["det_test_random"] or not args.det_test:
                        print("Episode_Count: ", item["det_test_random"], item["episode_counter"])
                        #episode_count = item.get("episode_count",0)
                        #if episode_count % log_interval == 0:
                        writer.add_scalar("charts/episodic_return", item["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", item["episode"]["l"], global_step)
                        # TramEnv additional info
                        writer.add_scalar("charts/cumsum_energy_consumption", item["episode"].get("cumsum_energy_consumption", 0), global_step)
                        writer.add_scalar("charts/pos_return", item["episode"].get("pos_return", 0), global_step)
                        writer.add_scalar("charts/velocity_return", item["episode"].get("velocity_return", 0), global_step)
                        writer.add_scalar("charts/schedule_return", item["episode"].get("schedule_return", 0), global_step)
                        writer.add_scalar("charts/energy_return", item["episode"].get("energy_return", 0), global_step)
                        break

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds].T)
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = pg_loss - (args.ent_coef*frac_ent) * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None:
                if approx_kl > args.target_kl:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("charts/entropy_coef", (args.ent_coef*frac_ent), global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    torch.save(agent.state_dict(), f"{directory}/agent.pt")
    if args.track:
    # make sure to tune `CHECKPOINT_FREQUENCY` 
    # so models are not saved too frequently
        torch.save(agent.state_dict(), f"{wandb.run.dir}/{args.exp_name}.pt")
        wandb.save(f"{wandb.run.dir}/agent.pt", policy="now")

    envs.close()
    writer.close()
    evaluator = Evaluator(run_name, directory, config_path, args.units)
    evaluator.evaluate_run()


