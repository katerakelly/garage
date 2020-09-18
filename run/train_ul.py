#!/usr/bin/env python3
"""This is an example to train a task with SAC algorithm written in PyTorch."""
import gym
import numpy as np
import click
import pickle as pkl
import torch
from torch import nn
from torch.nn import functional as F

from garage import wrap_experiment
from garage.experiment import deterministic, LocalRunner
from garage.replay_buffer import PathBuffer
from garage.sampler import LocalSampler
from garage.torch import set_gpu_mode
from garage.torch.modules import CNNEncoder
from garage.torch.algos import InverseMI, ULAlgorithm, CPC, RewardDecoder, ForwardMI
from garage.torch.modules import GaussianMLPTwoHeadedModule, MLPModule
from garage.misc.exp_util import make_env, make_exp_name


@click.command()
@click.option('--rb', default=None)
@click.option('--env', default='catcher-short')
@click.option('--algo', default=None)
@click.option('--image', is_flag=True)
@click.option('--discrete', is_flag=True)
@click.option('--name', default=None)
@click.option('--seed', default=1)
@click.option('--gpu', default=0)
@click.option('--debug', is_flag=True)
@click.option('--overwrite', is_flag=True)
def main(rb, env, algo, image, discrete, name, seed, gpu, debug, overwrite):
    name = make_exp_name(name, debug)
    if debug:
        overwrite = True # always allow overwriting on a debug exp
    @wrap_experiment(prefix=env, name=name, snapshot_mode='last', archive_launch_repo=False, use_existing_dir=overwrite)
    def train_inverse(ctxt, rb, env, algo, image, discrete, seed, gpu):
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
        env_name = env
        env = make_env(env_name, image, discrete)

        # make cnn encoder if learning from images
        cnn_encoder = None
        action_dim = env.spec.action_space.flat_dim
        obs_dim = env.spec.observation_space.flat_dim
        hidden_sizes = [256, 256]

        if image:
            print('Using IMAGE observations!')
            cnn_encoder = CNNEncoder(in_channels=1,
                                        output_dim=256)
            obs_dim = cnn_encoder.output_dim
            hidden_sizes = [] # linear decoder from conv features

        if 'inverse' in algo:
            # make mlp to predict actions
            action_mlp = MLPModule(input_dim=obs_dim * 2,
                                    output_dim=action_dim,
                                    hidden_sizes=hidden_sizes,
                                    hidden_nonlinearity=nn.ReLU)
            # pass inputs through CNN (if images), then though
            # mlp to predict actions
            if algo == 'inverse':
                predictors = {'InverseMI': InverseMI(cnn_encoder, action_mlp, discrete=discrete)}
            elif algo == 'inverse-reward':
                reward_mlp = MLPModule(input_dim=obs_dim,
                                        output_dim=1,
                                        hidden_sizes=hidden_sizes,
                                        hidden_nonlinearity=nn.ReLU)
                predictors = {'InverseMI': InverseMI(cnn_encoder, action_mlp, discrete=discrete), 'RewardDecode': RewardDecoder(cnn_encoder, reward_mlp)}
        elif algo == 'cpc':
            predictors = {'CPC': CPC(cnn_encoder)}
        elif algo == 'forward':
            predictors = {'ForwardMI': ForwardMI(cnn_encoder)}
        else:
            print('Algorithm {} not implemented.'.format(algo))
            raise NotImplementedError

        # load saved rb
        rb_filename = f'data/local/{env_name}/{rb}/replay_buffer.pkl'
        with open(rb_filename, 'rb') as f:
            replay_buffer = pkl.load(f)['replay_buffer']
        f.close()
        # test the obs to make sure it is what we expect
        test_obs = replay_buffer.sample_transitions(batch_size=1)['observation']
        assert len(test_obs.flatten()) == env.spec.observation_space.flat_dim

        # construct algo and train!
        algo = ULAlgorithm(predictors,
                           replay_buffer,
                           lr=1e-2,
                           buffer_batch_size=256)

        if torch.cuda.is_available():
            set_gpu_mode(True, gpu_id=gpu)
        else:
            set_gpu_mode(False)
        algo.to()
        runner.setup(algo=algo, env=env)
        runner.train(n_epochs=1000, batch_size=1000)

    train_inverse(rb=rb, env=env, algo=algo, image=image, discrete=discrete, seed=seed, gpu=gpu)

main()
