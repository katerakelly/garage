import os
import importlib
import numpy as np
import gym
from gym import spaces
from ple import PLE

from garage.envs.pygame.base import BaseEnv, ModifiedBaseEnv
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

    def __init__(self, discrete, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.quick_init(locals())
        self.discrete = discrete
        if not self.discrete:
            self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def step(self, action):
        # convert cont. action to discrete
        if not self.discrete:
            if action < 0.33:
                action = 0
            elif action > 0.33 and action < 0.66:
                action =1
            else:
                action = 2
        # policy outputs categorical dist over actions, take max
        else:
            action = np.argmax(action)
        return super().step(action)

    def render(self, mode='rgb_array', **kwargs):
        # don't pass other keyword args which are not supported
        return super().render(mode=mode)


class PygameCatcherShortEnv(PygameCatcherEnv, Serializable):
    """
    catcher game modified to end episode when fruit is caught
    """

    def __init__(self, discrete, *args, **kwargs):
        super().__init__(discrete, *args, **kwargs)
        self.quick_init(locals())

    def step(self, action):
        ob, reward, done, info = super().step(action)
        # end episode after catching attempt (success or failure)
        done = reward != 0
        return ob, reward, done, info


class PygameVariationShortEnv(ModifiedBaseEnv, Serializable):
    """
    modified catcher game to have a gripper on the agent
    also combines all the modifications of above envs
    """
    def __init__(self, game_name, discrete, normalize=True, display=False, **kwargs):
        self.game_name = game_name
        self.init(game_name, normalize, display, **kwargs)
        self.quick_init(locals())
        self.discrete = discrete
        if not self.discrete:
            self.action_space = spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32)

    def step(self, action):
        # convert cont. action to discrete
        if not self.discrete:
            bins = np.linspace(0, 1, num=len(self.action_set))
            action = sum(1 - (action < bins).astype(int)) - 1
        # policy outputs categorical dist over actions, take max
        else:
            action = np.argmax(action)
        ob, reward, done, info = super().step(action)
        # end episode after catching attempt (success or failure)
        done = reward != 0
        return ob, reward, done, info

    def render(self, mode='rgb_array', **kwargs):
        # don't pass other keyword args which are not supported
        return super().render(mode=mode)

    def get_ob_normalize(self, state):
        state_normal = self.get_ob(state)
        state_normal[0] = (state_normal[0] - 26) / 26
        state_normal[1] = (state_normal[1]) / 8
        state_normal[2] = (state_normal[2] - 26) / 26
        state_normal[3] = (state_normal[3] - 20) / 45
        return state_normal


class PygameGripperShortEnv(PygameVariationShortEnv, Serializable):
    def __init__(self, *args, **kwargs):
        self.quick_init(locals())
        super().__init__('gripper', *args, **kwargs)


class PygameArrowShortEnv(PygameVariationShortEnv, Serializable):
    def __init__(self, *args, **kwargs):
        self.quick_init(locals())
        super().__init__('arrow', *args, **kwargs)


    def step(self, action):
        ob, score, done, info = super().step(action)
        # compute a dense reward based on distance to fruit
        if score == 0:
            # HACK don't let this be 0, because 0 is reserved for not catching
            reward = min(-self.game.distance2fruit(), -1) / 100.
        elif score < 0:
            # agent just failed to catch fruit
            # this makes it identifiable
            reward = 0
        else:
            reward = score
        return ob, reward, done, info


