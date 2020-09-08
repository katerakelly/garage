#!/usr/bin/env python3
"""This is an example to train a task with SAC algorithm written in PyTorch."""
import gym
import numpy as np
import click
import torch
from torch import nn
from torch.nn import functional as F

from garage import wrap_experiment
from garage.experiment import deterministic, LocalRunner
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.algos import SAC
from garage.torch.modules import CNNEncoder
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction, ContinuousCNNQFunction
from garage.misc.exp_util import make_env, make_exp_name


@click.command()
@click.option('--env', default='cheetah')
@click.option('--image', is_flag=True)
@click.option('--name', default=None)
@click.option('--seed', default=1)
@click.option('--gpu', default=0)
@click.option('--debug', is_flag=True)
@click.option('--overwrite', is_flag=True)
def main(env, image, name, seed, gpu, debug, overwrite):
    name = make_exp_name(name, debug)
    if debug:
        overwrite = True # always allow overwriting on a debug exp
    @wrap_experiment(prefix=env, name=name, snapshot_mode='none', archive_launch_repo=False, use_existing_dir=overwrite)
    def sac_batch(ctxt, env, image, seed, gpu):
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

        # make cnn encoder if learning from images
        cnn_encoder = None
        input_dim = env.spec.observation_space.flat_dim + env.spec.action_space.flat_dim
        if image:
            print('Using IMAGE observations!')
            cnn_encoder = CNNEncoder(in_channels=1,
                                        output_dim=256)
            input_dim = cnn_encoder.output_dim + env.spec.action_space.flat_dim

        policy = TanhGaussianMLPPolicy(
            env_spec=env.spec,
            hidden_sizes=[256, 256],
            hidden_nonlinearity=nn.ReLU,
            output_nonlinearity=None,
            min_std=np.exp(-20.),
            max_std=np.exp(2.),
            cnn_encoder=cnn_encoder,
        )
        qf1_mlp = ContinuousMLPQFunction(env_spec=env.spec,
                                    input_dim=input_dim,
                                    hidden_sizes=[256, 256],
                                    hidden_nonlinearity=F.relu)
        qf1 = ContinuousCNNQFunction(cnn_encoder, qf1_mlp)

        qf2_mlp = ContinuousMLPQFunction(env_spec=env.spec,
                                    input_dim=input_dim,
                                    hidden_sizes=[256, 256],
                                    hidden_nonlinearity=F.relu)
        qf2 = ContinuousCNNQFunction(cnn_encoder, qf2_mlp)

        replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

        sac = SAC(env_spec=env.spec,
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
                reward_scale=100.,
                steps_per_epoch=1,
                cnn_encoder=cnn_encoder)

        if torch.cuda.is_available():
            set_gpu_mode(True, gpu_id=gpu)
        else:
            set_gpu_mode(False)
        sac.to()
        runner.setup(algo=sac, env=env, sampler_cls=LocalSampler)
        runner.train(n_epochs=1000, batch_size=1000)

    sac_batch(env=env, image=image, seed=seed, gpu=gpu)

main()
