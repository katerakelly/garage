import os
import importlib
import numpy as np
import gym
from gym import spaces
from ple import PLE

from garage.envs.pygame.base import BaseEnv
from garage.envs.serializable import Serializable


class CatcherEnv(BaseEnv):
  def __init__(self, normalize=True, display=False, **kwargs):
    self.game_name = 'Catcher'
    self.init(normalize, display, **kwargs)

  def get_ob_normalize(self, state):
    state_normal = self.get_ob(state)
    state_normal[0] = (state_normal[0] - 26) / 26
    state_normal[1] = (state_normal[1]) / 8
    state_normal[2] = (state_normal[2] - 26) / 26
    state_normal[3] = (state_normal[3] - 20) / 45
    return state_normal


class PygameCatcherEnv(CatcherEnv, Serializable):
    """
    stub wrapper for pygame environments
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quick_init(locals())
        self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def step(self, action):
        if action < 0.33:
            action = 0
        elif action > 0.33 and action < 0.66:
            action =1
        else:
            action = 2
        #action = np.argmax(action) # used for softmax action dist
        return super().step(action)
