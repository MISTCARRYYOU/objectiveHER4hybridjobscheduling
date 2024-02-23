from agent_DRL.utilities.OU_Noise import OU_Noise
from agent_DRL.exploration_strategies.Base_Exploration_Strategy import Base_Exploration_Strategy
import numpy as np


class OTD_Exploration(Base_Exploration_Strategy):
    """Ornstein-Uhlenbeck noise process exploration strategy"""
    def __init__(self, config):
        super().__init__(config)
        self.noise = OU_Noise(self.config.action_size, self.config.seed, self.config.hyperparameters["mu"],
                              self.config.hyperparameters["theta"], self.config.hyperparameters["sigma"])

    def perturb_action_for_exploration_purposes(self, action_info):
        """Perturbs the action of the agent to encourage exploration"""
        action_pre = action_info["action"]
        rate = action_info['rate']
        N = self.noise.sample()
        # print(N, 333333333)
        action = action_pre + N  # without action = action.clip(0,1)
        action = np.abs(action)
        for i in range(action.shape[0]):
            for j in range(action.shape[1]):
                if action[i][j] > 1.:
                    action[i][j] = action_pre[i][j]
        return action

    def add_exploration_rewards(self, reward_info):
        """Actions intrinsic rewards to encourage exploration"""
        raise ValueError("Must be implemented")

    def reset(self):
        """Resets the noise process"""
        self.noise.reset()
