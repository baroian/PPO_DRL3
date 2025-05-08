import argparse
import os
import distutils
import gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import time
from collections import deque 
import visuals # Import the plotting functions
import yaml # Import YAML

def parse_args():
    parser = argparse.ArgumentParser()
    # Keep non-config args first
    parser.add_argument('--exp-name', type=str, default=os.path.basename(__file__).rstrip(".py"),
        help='the name of this experiment')
    parser.add_argument('--cuda', type=lambda x: bool(distutils.util.strtobool(x)), default=True, nargs='?', const=True,
        help='if toggled, cuda will not be enabled by default')
    parser.add_argument('--config', type=str, default='ppo_config.yaml',
        help='path to the config file')

    # Define args that *can* be in YAML, but remove default= here
    parser.add_argument('--gym-id', type=str, help='the id of the gym environment')
    parser.add_argument('--learning-rate', type=float, help='the learning rate of the optimizer')
    parser.add_argument('--seed', type=int, help='seed of the experiment')
    parser.add_argument('--total-timesteps', type=int, help='total timesteps of the experiments')
    parser.add_argument('--torch-deterministic', type=lambda x: bool(distutils.util.strtobool(x)), nargs='?', const=True, help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--num-envs', type=int, help='the number of environments to run in parallel')
    parser.add_argument('--n-steps', type=int, help='the number of steps to run in each environment before updating the policy')
    parser.add_argument('--epochs', type=int, help='the number of epochs to run in each environment before updating the policy')
    parser.add_argument('--num-minibatches', type=int, help='the number of minibatches to split the batch into')
    parser.add_argument('--gamma', type=float, help='the discount factor gamma')
    parser.add_argument('--clip-coef', type=float, help='the clipping coefficient')
    parser.add_argument('--ent-coef', type=float, help='the entropy coefficient')
    parser.add_argument('--value-fn-coef', type=float, help='the value function coefficient')
    parser.add_argument('--max-grad-norm', type=float, help='the maximum gradient norm')
    parser.add_argument('--gae-lambda', type=float, help='the lambda parameter for GAE')
    # Load config file to set defaults 
    temp_args, _ = parser.parse_known_args()
    config_dict = {}
    if temp_args.config and os.path.exists(temp_args.config):
        with open(temp_args.config, 'r') as f:
            config_dict = yaml.safe_load(f)
        parser.set_defaults(**config_dict)

    args = parser.parse_args()

    # Calculate batch_size and minibatch_size 
    args.batch_size = args.num_envs * args.n_steps
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    return args


def make_env(gym_id, seed, idx):
    def thunk():
        env = gym.make(gym_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.reset(seed=seed + idx)
        return env
    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Agent(nn.Module):
    def __init__(self, envs):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, envs.single_action_space.n), std=0.01),
        )
    def get_value(self,x):
        return self.critic(x)

    def get_action_and_value(self,x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()  # pi(a|s)
        logpolicy = probs.log_prob(action) # log pi(a|s)
        return action, logpolicy, probs.entropy(), self.get_value(x)

if __name__ == "__main__":
    args = parse_args()
    print(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic


    # CHANGE WHAT GPU TO USE HERE
    device = torch.device("cuda:1" if torch.cuda.is_available() and args.cuda else "cpu")

    envs = gym.vector.SyncVectorEnv([make_env(args.gym_id, args.seed,i) for i in range(args.num_envs)])

    agent = Agent(envs).to(device)
    print(agent)

    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps = 1e-5)

    #  Data points recorded for each batch
    obs = torch.zeros((args.n_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.n_steps, args.num_envs) + envs.single_action_space.shape).to(device) 
    logpolicies = torch.zeros((args.n_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.n_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.n_steps, args.num_envs)).to(device)
    values = torch.zeros((args.n_steps, args.num_envs)).to(device)


    ### Initialize
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset()
    next_obs = torch.from_numpy(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // args.batch_size

    # store data for plotting
    all_update_steps, all_global_steps_ep_end, all_episodic_rewards, all_policy_losses, all_value_losses, all_entropy_losses, all_total_losses, all_explained_vars, all_clip_fracs, all_learning_rates = [], [], [], [], [], [], [], [], [], []

    for update in range(1, num_updates + 1):
        ### NOTE - we annealeate the learning rate 
        frac = 1.0 - (update / num_updates)  # such that it starts at 1 and goes to 0
        lrnow = frac * args.learning_rate
        optimizer.param_groups[0]['lr'] = lrnow
        # Store LR for plotting
        all_learning_rates.append(lrnow)
        all_update_steps.append(update)

        # clipfracs - measure how often the clipping occurs
        clipfracs = []

        for step in range(0, args.n_steps):
            global_step += 1*args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            ### Get action and value for current state from networks
            with torch.no_grad():
                action, logpolicy, entropy, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()

            ### Store action and logpolicy for current state
            actions[step] = action
            logpolicies[step] = logpolicy

            ### Step environment and store reward
            next_obs, reward, terminated, truncated, infos = envs.step(action.cpu().numpy())
            rewards[step] = torch.from_numpy(reward).to(device).view(-1)
            next_obs = torch.from_numpy(next_obs).to(device)
            next_done = torch.from_numpy(terminated).to(device)
            # collect rewards when episodes end
            if 'final_info' in infos:
                for info_dict in infos['final_info']:
                    if info_dict is not None and 'episode' in info_dict:
                        ep_rew = info_dict['episode']['r']
                        #print(f"Global step: {global_step} Episode reward: {ep_rew}")
                        all_global_steps_ep_end.append(global_step)
                        all_episodic_rewards.append(ep_rew)
                        break
          


        ### Compute returns and advantages
        with torch.no_grad():
            next_value = agent.get_value(next_obs).flatten()  # get estimated value of next state using network V(s')
            

            # Not GAE
            """
            returns = torch.zeros_like(rewards).to(device) #
            for t in reversed(range(args.n_steps)): # go backwards through the episode (backprop)
                if t == args.n_steps-1:
                    nextnonterminal = 1.0 - next_done.float() # 0 if done, 1 if not 
                    next_return = next_value
                else:
                    nextnonterminal = 1.0 - dones[t+1].float() 
                    next_return = returns[t+1]
                returns[t] = rewards[t] + args.gamma * nextnonterminal * next_return # discounted return

            # Not GAE - Generalized Advantage Estimation
            advantage = returns - values   #discounted return - estimated value 
            """

            # GAE - Generalized Advantage Estimation
            advantage = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.n_steps)):
                if t == args.n_steps-1:
                    nextnonterminal = 1.0 - next_done.float()
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t+1].float()
                    nextvalues = values[t+1]
                delta = rewards[t] + args.gamma * nextnonterminal * nextvalues - values[t]
                advantage[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantage + values


        ### flatten the batch data
        b_obs = obs.reshape(-1, *envs.single_observation_space.shape)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_logpolicies = logpolicies.reshape(-1)
        b_advantages = advantage.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        ### TRAINING 


        b_inds = np.arange(args.batch_size) # get the indices of the batch so we can shuffle them

        for epoch in range(args.epochs):
            np.random.shuffle(b_inds)

            for start in range(0, args.batch_size, args.minibatch_size): # go through the batch in minibatches
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]
                
                # forward pass - input is the indicies and the actions (already collected during rollout)
                _, newlogpolicy, entropy, new_values = agent.get_action_and_value(b_obs[mb_inds], b_actions.long()[mb_inds])

                logration = newlogpolicy - b_logpolicies[mb_inds]  # difference between the new and old policy
                ration = logration.exp() # r_t(theta)


                with torch.no_grad():
                    clipfracs += [((ration - 1.0).abs() > args.clip_coef).float().mean().item()]

                # Trick: advantage normalization
                mb_advantages = b_advantages[mb_inds]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8) # small value to not divide by 0

                # another trick - normalize the returns
                mb_returns = b_returns[mb_inds]
                mb_returns = (mb_returns - mb_returns.mean()) / (mb_returns.std() + 1e-8)
               
                # Policy Loss/objective
                objective = mb_advantages * ration  # r_t(theta) * advantage
                # clip the ration (diff between new and old policy)
                clipped_objective = mb_advantages * torch.clamp(ration, 1-args.clip_coef, 1+args.clip_coef)
                policy_loss = torch.min(objective, clipped_objective).mean()  

                # value loss - MSE between true and predicted value
                value_loss = 0.5 * ((new_values - mb_returns)**2).mean()  
                # note - we can also do value loss clipping  
                # note - without value loss clipping, the loss has spikes to 2000, so we're doing it
                #v_loss_unclipped = (new_values - b_returns[mb_inds])**2
                #v_clipped = b_values[mb_inds] + torch.clamp(new_values - b_values[mb_inds], -args.clip_coef, args.clip_coef)
                #v_loss_clipped = (v_clipped - b_returns[mb_inds])**2
                #value_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

                ### Entropy bonus
                entropy_loss = -entropy.mean()
                
                #final loss - minimize policy loss and value loss, maximize entropy 
                loss = -policy_loss + args.ent_coef * entropy_loss + args.value_fn_coef * value_loss

                ### backprop
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(),args.max_grad_norm)
                optimizer.step()
        

        # explained variance  - close to 1 means the critic predicts well the returns, can be - infinite 
        # to interpret (by normalizing) the quality of the critic over a batch
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        

        # store data for plotting
        all_policy_losses.append(policy_loss.item()); all_value_losses.append(value_loss.item()); all_entropy_losses.append(entropy_loss.item()); all_total_losses.append(loss.item()); all_clip_fracs.append(np.mean(clipfracs) if clipfracs else np.nan); all_explained_vars.append(explained_var)
        
        start_index = len(all_episodic_rewards) - args.batch_size
        avg_reward = np.mean(all_episodic_rewards[start_index:]) if all_episodic_rewards else np.nan
        if update % int(num_updates/10) == 0:
            print("update={} global_step={} avg_reward={:.2f} loss={:.4f} explained_var={:.4f} lr={:.2e}".format(update, global_step, avg_reward, loss.item(), explained_var, lrnow))

    # End of training loop



    # Generate Plots and Save Data
    visuals.plot_reward(all_global_steps_ep_end, all_episodic_rewards)
    visuals.plot_losses(all_update_steps, all_policy_losses, all_value_losses, all_entropy_losses, all_total_losses)
    visuals.plot_diagnostics(all_update_steps, all_explained_vars, all_clip_fracs, all_learning_rates)
    visuals.save_data(all_global_steps_ep_end, all_episodic_rewards, all_update_steps, all_total_losses)

    envs.close()
    # Save the arguments to a file 
    args_file = os.path.join("plots", "args.txt")
    with open(args_file, "w") as f:
        for arg, value in vars(args).items():
            f.write("{}: {}\n".format(arg, value))


    end_time = time.time()
    total_duration = end_time - start_time
    print("\nTotal training time: %.2f seconds" % total_duration)
