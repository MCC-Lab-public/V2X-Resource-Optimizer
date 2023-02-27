# -*- coding: utf-8 -*-
import gc
import logging
import os
import numpy as np
import csv
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

## AutoRL settings
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
import hpbandster.core.nameserver as hpns
from hpbandster.optimizers import randomsearch as RandomSearch
from hpbandster.optimizers import hyperband as HyperBand
from hpbandster.optimizers import BOHB as BOHB

# GPU usage
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
WEIGHT_FINAL_INIT = 3e-3
BIAS_FINAL_INIT = 3e-4

from matplotlib import pyplot as plt
import datetime as dt

from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import gym
from utils import OrnsteinUhlenbeckActionNoise
from replay_memory import ReplayMemory, Transition
from utils import NormalizedActions

from cell import Cell
from environment import CellularNetworksEnvironment

# System configurations
import json
with open('params.json', 'r') as f:
    params = json.load(f)
SEED_VALUE = 0
load_model = False
render_train = False
render_eval = False

logger = logging.getLogger('DDPG_Auto')
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Environment registration
from gym.envs.registration import register
register(
    id='cellularnetworks-v0',
    entry_point='environment:CellularNetworksEnvironment'
)

# AutoRL configuration space
DDPG_AUTO_CONFIGSPACE = {
    # NAS
    "hidden_layer_a1": CSH.UniformIntegerHyperparameter("hidden_layer_a1", lower=1, upper=1000, default_value=400, log=True),
    "hidden_layer_a2": CSH.UniformIntegerHyperparameter("hidden_layer_a2", lower=1, upper=1000, default_value=300, log=True),
    "hidden_layer_c1": CSH.UniformIntegerHyperparameter("hidden_layer_c1", lower=1, upper=1000, default_value=400, log=True),
    "hidden_layer_c2": CSH.UniformIntegerHyperparameter("hidden_layer_c2", lower=1, upper=1000, default_value=400, log=True),
    # HPO
    "lr_a": CSH.UniformFloatHyperparameter('lr_a', lower=1e-5, upper=1e-1, default_value=1e-3, log=True),
    "lr_c": CSH.UniformFloatHyperparameter('lr_c', lower=1e-5, upper=1e-1, default_value=1e-3, log=True),
    "wd_a": CSH.UniformFloatHyperparameter('wd_a', lower=0.0, upper=1e-1, default_value=1e-2, log=False),
    "wd_c": CSH.UniformFloatHyperparameter('wd_c', lower=0.0, upper=1e-1, default_value=1e-2, log=False),
    "tau": CSH.UniformFloatHyperparameter('tau', lower=0.0, upper=1.0, default_value=0.001, log=False),
    "gamma": CSH.UniformFloatHyperparameter('gamma', lower=0.0, upper=1.0, default_value=0.99, log=False),
    "target_update": CSH.UniformIntegerHyperparameter("target_update", lower=1, upper=20, default_value=10, log=True),
    "episode": CSH.UniformIntegerHyperparameter("episode", lower=1, upper=1000, default_value=500, log=True)
}

def fan_in_uniform_init(tensor, fan_in=None):
    # Utility function for initializing actor and critic
    if fan_in is None:
        fan_in = tensor.size(-1)

    w = 1.0 / np.sqrt(fan_in)
    nn.init.uniform_(tensor, -w, w)

def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_((1.0 - tau) * target_param.data + tau * param.data)

def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)

class Actor(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Actor, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_size[0])

        # Layer 2
        self.linear2 = nn.Linear(hidden_size[0], hidden_size[1])
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(hidden_size[1])

        # Output layer
        self.output = nn.Linear(hidden_size[1], num_outputs)

        # Weight initialization
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)
        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)
        nn.init.uniform_(self.output.weight, -WEIGHT_FINAL_INIT, WEIGHT_FINAL_INIT)
        nn.init.uniform_(self.output.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs):
        x = inputs
        # Layer 1
        x = self.linear1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # Layer 2
        x = self.linear2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # Output layer
        outputs = torch.tanh(self.output(x))
        return outputs

class Critic(nn.Module):
    def __init__(self, hidden_size, num_inputs, action_space):
        super(Critic, self).__init__()
        self.action_space = action_space
        num_outputs = action_space.shape[0]

        # Layer 1
        self.linear1 = nn.Linear(num_inputs, hidden_size[0])
        self.bn1 = nn.BatchNorm1d(hidden_size[0])

        # Layer 2
        self.linear2 = nn.Linear(hidden_size[0] + num_outputs, hidden_size[1])
        self.bn2 = nn.BatchNorm1d(hidden_size[1])

        # Output layer (single value)
        self.V = nn.Linear(hidden_size[1], 1)

        # Weight initialization
        fan_in_uniform_init(self.linear1.weight)
        fan_in_uniform_init(self.linear1.bias)
        fan_in_uniform_init(self.linear2.weight)
        fan_in_uniform_init(self.linear2.bias)
        nn.init.uniform_(self.V.weight, -WEIGHT_FINAL_INIT, WEIGHT_FINAL_INIT)
        nn.init.uniform_(self.V.bias, -BIAS_FINAL_INIT, BIAS_FINAL_INIT)

    def forward(self, inputs, actions):
        x = inputs
        # Layer 1
        x = self.linear1(x)
        x = self.bn1(x)
        x = F.relu(x)
        # Layer 2
        x = torch.cat((x, actions), 1) # Insert the actions
        x = self.linear2(x)
        x = self.bn2(x)
        x = F.relu(x)
        # Output
        V = self.V(x)
        return V

class DDPG(object):
    def __init__(self, config, num_inputs, action_space, checkpoint_dir=None):
        """
        Deep Deterministic Policy Gradient (https://arxiv.org/abs/1509.02971)
        :param gamma:           Discount factor
        :param tau:             Update factor for the actor and the critic
        :param hidden_size:     Number of units in the hidden layers of the actor and critic. (must be of length 2)
        :param num_inputs:      Size of the input states
        :param action_space:    The action space of the used environment.
                                Used to clip the actions and to distinguish the number of outputs
        :param checkpoint_dir:  Path as String to the directory to save the networks
                                If None then "./save_models/" will be used
        """
        self.action_space = action_space

        # Neural architecture search
        self.hidden_size_actor = [config['hidden_layer_a1'], config['hidden_layer_a2']]
        self.hidden_size_critic = [config['hidden_layer_c1'], config['hidden_layer_c2']]

        # Hyperparameter optimization
        self.lr_actor = config['lr_a']
        self.lr_critic = config['lr_c']
        self.wd_actor = config['wd_a']
        self.wd_critic = config['wd_c']
        self.gamma = config['gamma']
        self.tau = config['tau']
        self.target_update = config['target_update']

        # Define the actor
        self.actor = Actor(self.hidden_size_actor, num_inputs, self.action_space).to(dev)
        self.actor_target = Actor(self.hidden_size_actor, num_inputs, self.action_space).to(dev)

        # Define the critic
        self.critic = Critic(self.hidden_size_critic, num_inputs, self.action_space).to(dev)
        self.critic_target = Critic(self.hidden_size_critic, num_inputs, self.action_space).to(dev)

        # Define the optimizers for both networks
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor, weight_decay=self.wd_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr_critic, weight_decay=self.wd_critic)

        # Make sure both targets are with the same weight
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        # Set the directory to save the models
        if checkpoint_dir is None:
            self.checkpoint_dir = "./saved_models/"
        else:
            self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info('Saving all checkpoints to {}'.format(self.checkpoint_dir))

    def calc_action(self, state, action_noise=None):
        """
        Evaluates the action to perform in a given state
        :param state:           State to perform the action on in the env.
                                Used to evaluate the action.
        :param action_noise:    If not None, the noise to apply on the evaluated action
        :return:
        """
        x = state.to(dev)

        # Get the continuous action value to perform in the env
        self.actor.eval()
        mu = self.actor(x)
        self.actor.train()
        mu = mu.data

        # During training we add noise for exploration
        if action_noise is not None:
            noise = torch.Tensor(action_noise.noise()).to(dev)
            mu += noise

        # Clip the output according to the action space of the env
        mu = mu.clamp(self.action_space.low[0], self.action_space.high[0])

        return mu

    def update_params(self, batch):
        """
        Updates the parameters/networks of the agent according to the given batch.
            1. Compute the target
            2. Update the Q-function/critic by one step of gradient descent
            3. Update the policy/actor by one step of gradient ascent
            4. Update the target networks through a soft update
        :param batch:   Batch to perform the training of the parameters
        :return:
        """
        # Get tensors from the batch
        state_batch = torch.cat(batch.state).to(dev)
        action_batch = torch.cat(batch.action).to(dev)
        reward_batch = torch.cat(batch.reward).to(dev)
        done_batch = torch.cat(batch.done).to(dev)
        next_state_batch = torch.cat(batch.next_state).to(dev)

        # Get the actions and the state values to compute the targets
        next_action_batch = self.actor_target(next_state_batch)
        next_state_action_values = self.critic_target(next_state_batch, next_action_batch.detach())

        # Compute the target
        reward_batch = reward_batch.unsqueeze(1)
        done_batch = done_batch.unsqueeze(1)
        expected_values = reward_batch + (1.0 - done_batch) * self.gamma * next_state_action_values

        # Clipping the expected values here?
        # expected_values = torch.clamp(expected_values, min_value, max_value)

        # Update the critic networks
        self.critic_optimizer.zero_grad()
        state_action_batch = self.critic(state_batch, action_batch)
        value_loss = F.mse_loss(state_action_batch, expected_values.detach())
        value_loss.backward()
        self.critic_optimizer.step()

        # Update the actor networks
        self.actor_optimizer.zero_grad()
        policy_loss = self.critic(state_batch, self.actor(state_batch))
        policy_loss = -policy_loss.mean()
        policy_loss.backward()
        self.actor_optimizer.step()

        # Update the target networks
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)

        return value_loss.item(), policy_loss.item()

    def save_checkpoint(self, last_timestep, replay_buffer):
        """
        Saving the networks and all parameters to a file in 'checkpoint_dir'
        :param last_timestep:  Last timestep in training before saving
        :param replay_buffer:   Current replay buffer
        :return:
        """
        checkpoint_name = self.checkpoint_dir + '/ep_{}.pth.tar'.format(last_timestep)
        logger.info('Saving checkpoint...')
        checkpoint = {
            'last_timestep': last_timestep,
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'replay_buffer': replay_buffer
        }
        logger.info('Saving model at timestep {}...'.format(last_timestep))
        torch.save(checkpoint, checkpoint_name)
        gc.collect()
        logger.info('Saved model at timestep {} to {}'.format(last_timestep, self.checkpoint_dir))

    def get_path_of_latest_file(self):
        """
        Returns the latest created file in 'checkpoint_dir'
        """
        files = [file for file in os.listdir(self.checkpoint_dir) if (file.endswith(".pt") or file.endswith(".tar"))]
        filepaths = [os.path.join(self.checkpoint_dir, file) for file in files]
        last_file = max(filepaths, key=os.path.getctime)
        return os.path.abspath(last_file)

    def load_checkpoint(self, checkpoint_path=None):
        """
        Saving the networks and all parameters from a given path.
        If the given path is None, then the latest saved file in 'checkpoint_dir' will be used.
        :param checkpoint_path: File to load the model from
        :return:
        """
        if checkpoint_path is None:
            checkpoint_path = self.get_path_of_latest_file()

        if os.path.isfile(checkpoint_path):
            logger.info('Loading checkpoint...({})'.format(checkpoint_path))
            key = 'cuda' if torch.cuda.is_available() else 'cpu'

            checkpoint = torch.load(checkpoint_path, map_location=key)
            start_timestep = checkpoint['last_timestep'] + 1
            self.actor.load_state_dict(checkpoint['actor'])
            self.critic.load_state_dict(checkpoint['critic'])
            self.actor_target.load_state_dict(checkpoint['actor_target'])
            self.critic_target.load_state_dict(checkpoint['critic_target'])
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
            replay_buffer = checkpoint['replay_buffer']

            gc.collect()
            logger.info('Loaded model at timestep {} from {}'.format(start_timestep, checkpoint_path))
            return start_timestep, replay_buffer
        else:
            raise OSError('Checkpoint not found')

    def set_eval(self):
        """
        Sets the model in evaluation mode
        :return:
        """
        self.actor.eval()
        self.critic.eval()
        self.actor_target.eval()
        self.critic_target.eval()

    def set_train(self):
        """
        Sets the model in training mode
        :return:
        """
        self.actor.train()
        self.critic.train()
        self.actor_target.train()
        self.critic_target.train()

    def get_network(self, name):
        if name == 'Actor':
            return self.actor
        elif name == 'Critic':
            return self.critic
        else:
            raise NameError('name \'{}\' is not defined as a network'.format(name))

class AutoRLWorker(Worker):
    def __init__(self, **kwargs):
        super(AutoRLWorker, self).__init__(**kwargs)
        # Define the directory where to save and load models
        self.ckpt_dir = params['savedir'] + params['env']
        self.writer = SummaryWriter('runs/run_test1')

        # Define the reward threshold when the task is solved (if existing) for model saving
        self.reward_threshold = np.inf
        self.auto_rl_cnt = 0

    def compute(self, config, budget, working_directory, *args, **kwargs):
        print("Budget: {}".format(budget))
        self.auto_rl_cnt += 1
        final_reward = 0
        final_policy_loss = None
        final_value_loss = None

        # Create the env
        env = gym.make('cellularnetworks-v0')
        self.env = NormalizedActions(env)

        # Define DDPG agent
        agent = DDPG(config, self.env.observation_space.shape[0], self.env.action_space, checkpoint_dir=self.ckpt_dir)

        # Initialize replay memory
        memory = ReplayMemory(int(params['REPLAY_SIZE']))

        nb_actions = self.env.action_space.shape[-1]
        ou_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(nb_actions), sigma=float(params['NOISE_STDDEV']) * np.ones(nb_actions))

        # Define counters and other variables
        start_step = 0
        step = start_step // 10000 + 1
        rewards, policy_losses, value_losses, mean_test_rewards = [], [], [], []
        epoch = 0
        t = 0
        time_last_ckpt = time.time()

        # Start training
        logger.info('DOING {} TIMESTEPS'.format(params['TIMESTEPS']))
        logger.info("START AT TIMESTEP {0} WITH t = {1}".format(step, t))
        logger.info("START TRAINING AT {}".format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

        self.env.set_episode(config['episode'])

        for step in range(int(budget)):
            ou_noise.reset()
            epoch_return = 0.0
            epoch_value_loss = 0.0
            epoch_policy_loss = 0.0

            state = torch.Tensor(np.array([self.env.reset()])).to(dev)
            cnt = 1

            while True:
                if render_train:
                    self.env.render()

                action = agent.calc_action(state, ou_noise)
                next_state, reward, done, _ = self.env.step(action.cpu().numpy()[0])
                step += 1
                epoch_return += reward

                mask = torch.Tensor(np.array([done])).to(dev)
                reward = torch.Tensor(np.array([reward])).to(dev)
                next_state = torch.Tensor(np.array([next_state])).to(dev)

                memory.push(state, action, mask, next_state, reward)

                state = next_state

                if len(memory) > params['BATCH_SIZE']:
                    transitions = memory.sample(params['BATCH_SIZE'])
                    batch = Transition(*zip(*transitions))

                    value_loss, policy_loss = agent.update_params(batch)

                    epoch_value_loss += value_loss
                    epoch_policy_loss += policy_loss

                if done:
                    break

            epoch_return /= params['EPI']
            epoch_value_loss /= params['EPI']
            epoch_policy_loss /= params['EPI']

            # env.render()
            rewards.append(epoch_return)
            value_losses.append(epoch_value_loss)
            policy_losses.append(epoch_policy_loss)
            self.writer.add_scalar('epoch/mean reward', epoch_return, epoch)
            self.writer.add_scalar('epoch/mean value loss', epoch_value_loss, epoch)
            self.writer.add_scalar('epoch/mean policy loss', epoch_policy_loss, epoch)
            if epoch_return >= final_reward:
                final_reward = epoch_return
                final_value_loss = epoch_value_loss
                final_policy_loss = epoch_policy_loss
            with open('./log/learning_status_' + str(self.auto_rl_cnt) + '.csv', 'a+', encoding='utf-8', newline='') as ptr:
                wr = csv.writer(ptr)
                wr.writerow([epoch, epoch_return, epoch_value_loss, epoch_policy_loss])

            # Test every 10-th episode (== 10,000) steps for a number of test_epochs epochs
            if step >= 10000 * t:
                t += 1
                test_rewards = []
                for _ in range(params["N_TEST_CYCLES"]):
                    state = torch.Tensor(np.array([self.env.reset()])).to(dev)
                    test_reward = 0
                    while True:
                        if render_eval:
                            self.env.render()

                        action = agent.calc_action(state)  # Selection without noise

                        next_state, reward, done, _ = self.env.step(action.cpu().numpy()[0])
                        test_reward += reward

                        next_state = torch.Tensor(np.array([next_state])).to(dev)

                        state = next_state
                        if done:
                            break
                    test_rewards.append(test_reward)

                mean_test_rewards.append(np.mean(test_rewards))

                for name, param in agent.actor.named_parameters():
                    self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)
                for name, param in agent.critic.named_parameters():
                    self.writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

                self.writer.add_scalar('test/mean_test_return', mean_test_rewards[-1], epoch)
                logger.info(
                    'EPOCH: {}, CURRENT TIMESTEP: {}, LAST REWARD: {}, MEAN REWARD: {}, MEAN TEST REWARD: {}'.format(
                        epoch, step, rewards[-1], np.mean(rewards[-10:]), np.mean(test_rewards)))

                # Save if the mean of the last three averaged rewards while testing is greater than the specified reward threshold
                if np.mean(mean_test_rewards[-3:]) >= self.reward_threshold:
                    agent.save_checkpoint(step, memory)
                    time_last_checkpoint = time.time()
                    logger.info(
                        'SAVED MODEL AT {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))

            epoch += 1

        curr_config = [self.auto_rl_cnt,
                       config['hidden_layer_a1'], config['hidden_layer_a2'], config['hidden_layer_c1'],
                       config['hidden_layer_c2'],
                       config['lr_a'], config['lr_c'], config['wd_a'], config['wd_c'],
                       config['tau'], config['gamma'], config['target_update'], config['episode'],
                       final_reward, final_value_loss, final_policy_loss]
        print("Auto RL iter #{}".format(self.auto_rl_cnt))
        print("Hidden layers (Actor: [{}, {}] / Critic: [{}, {}])".format(config['hidden_layer_a1'],
                                                                          config['hidden_layer_a2'],
                                                                          config['hidden_layer_c1'],
                                                                          config['hidden_layer_c2']))
        print("Learning rate (Actor: {} / Critic: {})".format(config['lr_a'], config['lr_c']))
        print("Weight decay (Actor: {} / Critic: {})".format(config['wd_a'], config['wd_c']))
        print("Tau: {} / Gamma: {}".format(config['tau'], config['gamma']))
        print("Update period: {} / Episode: {}".format(config['target_update'], config['episode']))
        print("Reward: {} / Value loss: {} / Policy loss: {}".format(final_reward, final_value_loss, final_policy_loss))
        with open('./log/configurations.csv', 'a+', encoding='utf-8', newline='') as ptr:
            wr = csv.writer(ptr)
            wr.writerow(curr_config)

        agent.save_checkpoint(step, memory)
        logger.info('SAVED MODEL AT ENDTIME {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
        logger.info('STOPPING TRAINING AT {}'.format(time.strftime('%a, %d %b %Y %H:%M:%S GMT', time.localtime())))
        self.env.close()
        return ({
            'loss': -final_reward, # AutoRL minimize it!
            'info': {'final_reward': final_reward,
                     'final_value_loss': final_value_loss,
                     'final_policy_loss': final_policy_loss
                     }
        })

    @staticmethod
    def get_configspace():
        """
        It builds the configuration space with the needed hyperparameters.
        It is easily possible to implement different types of hyperparameters.
        Beside float-hyperparameters on a log scale, it is also able to handle categorical input parameter.
        :return: ConfigurationsSpace-Object
        """
        cs = CS.ConfigurationSpace()
        ## NAS
        cs.add_hyperparameters([DDPG_AUTO_CONFIGSPACE["hidden_layer_a1"], DDPG_AUTO_CONFIGSPACE["hidden_layer_a2"], DDPG_AUTO_CONFIGSPACE["hidden_layer_c1"], DDPG_AUTO_CONFIGSPACE["hidden_layer_c2"]])
        ## HPO (float)
        cs.add_hyperparameters([DDPG_AUTO_CONFIGSPACE["lr_a"], DDPG_AUTO_CONFIGSPACE["lr_c"], DDPG_AUTO_CONFIGSPACE["wd_a"], DDPG_AUTO_CONFIGSPACE["wd_c"], DDPG_AUTO_CONFIGSPACE["tau"], DDPG_AUTO_CONFIGSPACE["gamma"]])
        ## HPO (integer)
        cs.add_hyperparameters([DDPG_AUTO_CONFIGSPACE["target_update"], DDPG_AUTO_CONFIGSPACE["episode"]])

        return cs


if __name__ == "__main__":
    min_budget = 10
    max_budget = 200
    NS = hpns.NameServer(run_id='0', host='127.0.0.1', port=None)
    NS.start()

    worker = AutoRLWorker(run_id='0', nameserver='127.0.0.1')
    worker.run(background=True)

    # randomsearch = RandomSearch(configspace=worker.get_configspace(), min_budget=min_budget, max_budget=max_budget)
    # res = randomsearch.run(n_iterations=4)
    # hyperband = HyperBand(configspace=worker.get_configspace(), min_budget=min_budget, max_budget=max_budget)
    # res = hyperband.run(n_iterations=4)
    bohb = BOHB(configspace=worker.get_configspace(), run_id='0', nameserver='127.0.0.1', min_budget=min_budget, max_budget=max_budget)
    res = bohb.run(n_iterations=10)

    # cs = worker.get_configspace()

    ## Shutdown
    # randomsearch.shutdown(shutdown_workers=True)
    # hyperband.shutdown(shutdown_workers=True)
    bohb.shutdown(shutdown_workers=True)
    NS.shutdown()

    ## Analysis
    id2config = res.get_id2config_mapping()
    incumbent = res.get_incumbent_id()

    print('Best found configuration:', id2config[incumbent]['config'])
    print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
    print('A total of %i runs where executed.' % len(res.get_all_runs()))
    print('Total budget corresponds to %.1f full function evaluations.' % (sum([r.budget for r in res.get_all_runs()]) / max_budget))