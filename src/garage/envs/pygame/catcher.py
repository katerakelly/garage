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

    def step(self, action):
        action = np.argmax(action)
        return super().step(action)
