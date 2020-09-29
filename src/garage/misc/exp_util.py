import os
import glob
import gym
import numpy as np
import datetime
import dateutil.tz

from garage.envs import GarageEnv, normalize
from garage.envs.wrappers import Grayscale, Resize, StackFrames, PixelObservationWrapper
from natural_rl_environment.natural_env import ReplaceBackgroundEnv
from natural_rl_environment.matting import BackgroundMattingWithColor
from natural_rl_environment.imgsource import RandomColorSource, RandomVideoSource


def make_env(env_name, is_image, bg=None, discrete=False, frame_stack=1):
    # error checking
    if frame_stack == 1 and 'cheetah' in env_name:
        print('this env needs velocity information')
        raise Exception
    if discrete and 'cheetah' in env_name:
        print('cheetah is not a discrete env')
        raise Exception

    if env_name == 'cheetah':
        env = gym.make('HalfCheetah-v2')
    elif env_name == 'catcher':
        env = gym.make('Catcher-PLE-serial-v0', discrete=discrete)
    elif env_name == 'catcher-short':
        env = gym.make('Catcher-PLE-serial-short-v0', discrete=discrete)
    elif env_name == 'gripper-short':
        env = gym.make('Gripper-PLE-serial-short-v0', discrete=discrete)
    elif env_name == 'arrow-short':
        env = gym.make('Arrow-PLE-serial-short-v0', discrete=discrete, delay=6)
    elif env_name == 'arrow-short-d0':
        env = gym.make('Arrow-PLE-serial-short-v0', discrete=discrete, delay=0)
    if is_image:
        env = PixelObservationWrapper(env)
        if bg == 'ideal_gas':
            print('Adding ideal gas distractors to background of images')
            g = '/home/rakelly/garage/distractors/*.mp4'
            files = glob.glob(os.path.expanduser(g))
            imgsource = RandomVideoSource((64, 64), files)
            env = ReplaceBackgroundEnv(env, BackgroundMattingWithColor((0, 0, 0)), imgsource)
        env = Grayscale(env)
        env = Resize(env, 64, 64)
        if frame_stack > 1:
            env = StackFrames(env, frame_stack)
        env = GarageEnv(env, is_image=is_image)
        env = normalize(env, normalize_obs=True, image_obs=True, flatten_obs=True)
    else:
        return GarageEnv(normalize(env))
    return env

def make_exp_name(name, debug):
    # name exps by date and time if no name is given
    if name is None:
        if debug:
            name = 'debug'
        else:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            name = now.strftime('%m_%d_%H_%M_%S')
    return name

