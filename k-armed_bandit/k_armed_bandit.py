import numpy as np
from typing import List, Dict, Optional
import gym
from gym import spaces
from gym.utils import seeding


class BanditEnv(gym.Env):
    """
    Bandit environment base to allow agents to interact with the class n-armed bandit
    in different variations

    p_dist:
        A list of probabilities of the likelihood that a particular bandit will pay out
    r_dist:
        A list of either rewards (if number) or means and standard deviations (if list)
        of the payout that bandit has
    """
    def __init__(self, bandits: int):
        self.seed = self._seed()
        self.bandits = bandits
        self.p_dist = np.full(bandits, 1)
        self.walk_dist = None
        self.set_r_dist(np.repeat([[0.0, 1.0]], bandits, axis=0))
        self.action_space = spaces.Discrete(bandits)
        self.observation_space = spaces.Discrete(0)

    def _seed(self, seed: Optional[int] = None) -> List[int]:
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_r_dist(self, r_dist: object):
        self.r_dist = r_dist
        if self.walk_dist is not None:
            self.r_dist_orig = r_dist.copy()

    def step(self, action: int) -> (int, float, bool, Dict[object, object]):
        reward = 0.0
        done = False

        # get the reward
        if self.np_random.uniform() <= self.p_dist[action]:
            reward = self.np_random.normal(*self.r_dist[action])

        # walk the means of payouts
        if self.walk_dist is not None:
            for r in range(len(self.r_dist)):
                self.r_dist[r][0] += (
                    self.np_random.normal(*self.walk_dist[r]))

        return 0, reward, done, {}

    def reset(self) -> int:
        if self.walk_dist is not None:
            self.r_dist = self.r_dist_orig.copy()

        return 0

    def render(self, mode='human', close=False):
        pass
