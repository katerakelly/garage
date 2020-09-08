import gym
import numpy as np
import datetime
import dateutil.tz

from garage.envs import GarageEnv, normalize
from garage.envs.wrappers import Grayscale, Resize, StackFrames, PixelObservationWrapper


def make_env(env_name, is_image, frame_stack):
    if env_name == 'cheetah':
        env = gym.make('HalfCheetah-v2')
    elif env_name == 'catcher':
        env = gym.make('Catcher-PLE-serial-v0')
    elif env_name == 'catcher-short':
        env = gym.make('Catcher-PLE-serial-short-v0')
    if is_image:
        env = PixelObservationWrapper(env)
        env = Grayscale(env)
        env = Resize(env, 64, 64)
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

