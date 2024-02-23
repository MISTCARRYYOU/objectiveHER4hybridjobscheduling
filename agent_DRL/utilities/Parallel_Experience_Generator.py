import random
import torch
import sys
from contextlib import closing
from utilities import record_env_variables
#
# from pathos.multiprocessing import ProcessingPool as Pool

from torch.multiprocessing import Pool
from random import randint
import pandas as pd
import numpy as np
import time
from agent_DRL.utilities.OU_Noise import OU_Noise
from agent_DRL.utilities.Utility_Functions import create_actor_distribution
# from agent_DRL.args import is_have_a_look

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


class Parallel_Experience_Generator(object):

    """ Plays n episode in parallel using a fixed agent. Only works for PPO or DDPG type agents at the moment, not Q-learning agents"""

    def __init__(self, environment, policy, seed, hyperparameters, action_size, use_GPU=False, action_choice_output_columns=None, config=None):
        self.use_GPU = use_GPU
        self.environment = environment
        self.action_types = "DISCRETE" if self.environment.action_space.dtype in [int, 'int64'] else "CONTINUOUS"
        self.action_size = action_size
        self.policy = policy
        self.action_choice_output_columns = action_choice_output_columns
        self.hyperparameters = hyperparameters
        if self.action_types == "CONTINUOUS": self.noise = OU_Noise(self.action_size, seed, self.hyperparameters["mu"],
                            self.hyperparameters["theta"], self.hyperparameters["sigma"])

        self.use_server =  False
        self.config = config

    def play_n_episodes(self, n, exploration_epsilon=None, episode_num=None, use_server=None):
        """Plays n episodes in parallel using the fixed policy and returns the data"""
        self.exploration_epsilon = exploration_epsilon
        if use_server is not None:
            self.use_server = use_server
        if episode_num is not None:
            self.episode_num = episode_num
        with closing(Pool(processes=n)) as pool:
            results = pool.map(self, range(n))
            pool.terminate()
        states_for_all_episodes = [episode[0] for episode in results]
        actions_for_all_episodes = [episode[1] for episode in results]
        rewards_for_all_episodes = [episode[2] for episode in results]
        return states_for_all_episodes, actions_for_all_episodes, rewards_for_all_episodes

    def __call__(self, n):
        exploration = max(0.0, random.uniform(self.exploration_epsilon / 3.0, self.exploration_epsilon * 3.0))
        return self.play_1_episode(exploration)

    def play_1_episode(self, epsilon_exploration):
        """Plays 1 episode using the fixed policy and returns the data"""
        state = self.reset_game()
        done = False
        episode_states = []
        episode_actions = []
        episode_rewards = []
        t1 = time.time()
        while not done:
            action = self.pick_action(self.policy, state, epsilon_exploration)
            next_state, reward, done, _ = self.environment.step(action)
            if self.hyperparameters["clip_rewards"]: reward = max(min(reward, 1.0), -1.0)
            if self.use_server == 0:
                pass
            episode_states.append(state)
            episode_actions.append(action)
            episode_rewards.append(reward)
            state = next_state

            if self.episode_num % self.config.record_interval == 0:
                with open('./my_data_and_graph/' + self.environment.env_name + '_trans_' + 'PPO' + '.txt',
                          'a') as f:
                    print('done/reward/state: ', done, reward, next_state.tolist(), file=f)
                    print('action: ', action.tolist(), file=f)
                    print('env variables: ', record_env_variables(self.environment), file=f)

        # with open('./my_data_and_graph/DRLs/' + 'PPO' + '.txt', 'a') as f:
        #     print('episode: {}  episoder reward: {}  episode cost: {}, [time-scaled, time, expenditure-scaled,expenditure, reliability-scaled, reliability]: {}'.format(
        #         self.episode_num, sum(episode_rewards), temp_cost, temp_cost_detail), file=f)
            # print('random action:', self.random_action, 'deter action:', self.determinate_action, file=f)

        pass
        return episode_states, episode_actions, episode_rewards

    def reset_game(self):
        """Resets the game environment so it is ready to play a new episode"""
        seed = randint(0, sys.maxsize)
        torch.manual_seed(seed) # Need to do this otherwise each worker generates same experience
        state = self.environment.reset()
        if self.action_types == "CONTINUOUS": self.noise.reset()

        self.random_action = 0
        self.determinate_action = 0
        return state

    def pick_action(self, policy, state, epsilon_exploration=None):
        """Picks an action using the policy"""
        if self.action_types == "DISCRETE":
            action_mask = self.environment.get_mask()
            if not self.config.evaluate:
                if random.random() <= epsilon_exploration:
                    avail_actions = (action_mask == False).nonzero().squeeze()
                    if len(avail_actions.shape) == 0:
                        action = avail_actions.item()
                    else:
                        action = np.random.choice(avail_actions, 1, replace=False)[0]
                    self.random_action += 1
                    return action

        state = torch.from_numpy(state).float().unsqueeze(0)
        actor_output = policy.forward(state)

        if self.action_choice_output_columns is not None:
            actor_output = actor_output[:, self.action_choice_output_columns]
        action_distribution = create_actor_distribution(self.action_types, actor_output, self.action_size)
        action_pre = action_distribution.sample().cpu()

        if self.action_types == "CONTINUOUS":
            action = action_pre + torch.Tensor(self.noise.sample())
            action = torch.abs(action)
            for i in range(action.shape[0]):
                if action[i] > 1.:
                    action[i] = 1.
        else: action = action_pre.item()
        self.determinate_action += 1
        return action
