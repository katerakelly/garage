#!/usr/bin/env python3
""" Collect and save data from a random or trained policy """
import gym
import numpy as np
import click
import datetime
import dateutil.tz
import torch
from torch import nn
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import GarageEnv, normalize
from garage.envs.wrappers import Grayscale, Resize, StackFrames, PixelObservationWrapper
from garage.experiment import deterministic, LocalRunner
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.np.policies import RandomPolicy
from garage.torch.algos import DataCollector

def make_env(env_name, is_image):
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
        env = StackFrames(env, 4)
        env = GarageEnv(env, is_image=is_image)
        env = normalize(env, normalize_obs=True, image_obs=True, flatten_obs=True)
    else:
        return GarageEnv(normalize(env))
    return env


@click.command()
@click.option('--env', default='cheetah')
@click.option('--image', is_flag=True)
@click.option('--name', default=None)
@click.option('--seed', default=1)
@click.option('--gpu', default=0)
@click.option('--debug', is_flag=True)
def main(env, image, name, seed, gpu, debug):
    # name exps by date and time if no name is given
    if name is None:
        if debug:
            name = 'debug'
        else:
            now = datetime.datetime.now(dateutil.tz.tzlocal())
            name = now.strftime('%m_%d_%H_%M_%S')

    @wrap_experiment(prefix=env, name=name, snapshot_mode='last', archive_launch_repo=False)
    def collect_data(ctxt, env, image, seed, gpu):
        """Set up environment and algorithm and run the task.

        Args:
            ctxt (garage.experiment.ExperimentContext): The experiment
                configuration used by LocalRunner to create the snapshotter.
            seed (int): Used to seed the random number generator to produce
                determinism.

        """
        deterministic.set_seed(seed)
        runner = LocalRunner(snapshot_config=ctxt)

        # make the env, given name and whether to use image obs
        env = make_env(env, image)

        policy = RandomPolicy(env_spec=env.spec)

        replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

        algo = DataCollector(policy, replay_buffer, steps_per_epoch=1, max_path_length=500)

        """
        if torch.cuda.is_available():
            set_gpu_mode(True, gpu_id=gpu)
        else:
            set_gpu_mode(False)
        """
        #algo.to()
        runner.setup(algo=algo, env=env, sampler_cls=LocalSampler)
        runner.train(n_epochs=100, batch_size=1000) # add arg store_paths=True to store collected samples. samples from each iter will be saved by snapshotter, but will be organized per itr collected. can we instead add the replay buffer to the list of things to be snapshotted?

    collect_data(env=env, image=image, seed=seed, gpu=gpu)

main()
