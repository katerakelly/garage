#!/usr/bin/env python3
"""This is an example to train a task with SAC algorithm written in PyTorch."""
import gym
import numpy as np
import click
import torch
from torch import nn
from torch.nn import functional as F

from garage import wrap_experiment
from garage.envs import GarageEnv, normalize
from garage.experiment import deterministic, LocalRunner
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import DiscreteSAC
from garage.torch.policies import CategoricalMLPPolicy
from garage.torch.q_functions import DiscreteMLPQFunction


@click.command()
@click.option('--env', default='cheetah')
@click.option('--image', is_flag=True)
@click.option('--seed', default=1)
@wrap_experiment(snapshot_mode='none', archive_launch_repo=False)
def sac_half_cheetah_batch(ctxt, env, image, seed):
    """Set up environment and algorithm and run the task.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        env (str): shorthand for env to use
        image (bool): whether to use image obs or underlying state

    """
    deterministic.set_seed(seed)
    runner = LocalRunner(snapshot_config=ctxt)
    if env == 'cheetah':
        env = GarageEnv(normalize(gym.make('HalfCheetah-v2')), is_image=image)
    elif env == 'catcher':
        env = GarageEnv(normalize(gym.make('Catcher-PLE-serial-v0')))

    policy = CategoricalMLPPolicy(
        env_spec=env.spec,
        hidden_sizes=[256, 256],
        hidden_nonlinearity=nn.ReLU,
    )

    qf1 = DiscreteMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)

    qf2 = DiscreteMLPQFunction(env_spec=env.spec,
                                 hidden_sizes=[256, 256],
                                 hidden_nonlinearity=F.relu)

    replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

    sac = DiscreteSAC(env_spec=env.spec,
              policy=policy,
              qf1=qf1,
              qf2=qf2,
              gradient_steps_per_itr=1000,
              max_path_length=500,
              replay_buffer=replay_buffer,
              min_buffer_size=1e4,
              target_update_tau=5e-3,
              discount=0.99,
              buffer_batch_size=256,
              reward_scale=10.,
              steps_per_epoch=1)

    if torch.cuda.is_available():
        set_gpu_mode(True)
    else:
        set_gpu_mode(False)
    sac.to()
    runner.setup(algo=sac, env=env, sampler_cls=LocalSampler)
    runner.train(n_epochs=1000, batch_size=1000)


sac_half_cheetah_batch()