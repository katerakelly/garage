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
from garage.torch.algos import DiscreteSAC
from garage.torch.modules import CNNEncoder
from garage.torch.policies import TanhGaussianMLPPolicy
from garage.torch.policies import CategoricalMLPPolicy
from garage.torch.q_functions import ContinuousMLPQFunction, CompositeQFunction, DiscreteMLPQFunction
from garage.misc.exp_util import make_env, make_exp_name


@click.command()
@click.option('--env', default='catcher')
@click.option('--image', is_flag=True)
@click.option('--discrete', is_flag=True)
@click.option('--name', default=None)
@click.option('--seed', default=1)
@click.option('--gpu', default=0)
@click.option('--debug', is_flag=True)
@click.option('--overwrite', is_flag=True)
@click.option('--pretrain', default=None)
@click.option('--train_cnn', is_flag=True)
def main(env, image, discrete, name, seed, gpu, debug, overwrite, pretrain, train_cnn):
    name = make_exp_name(name, debug)
    if debug:
        overwrite = True # always allow overwriting on a debug exp
    @wrap_experiment(prefix=env, name=name, snapshot_mode='none', archive_launch_repo=False, use_existing_dir=overwrite)
    def sac_batch(ctxt, env, image, discrete, pretrain, train_cnn, seed, gpu):
        """Set up environment and algorithm and run the task.

        Args:
            ctxt (garage.experiment.ExperimentContext): The experiment
                configuration used by LocalRunner to create the snapshotter.
            env (str): shorthand for which env to create
            discrete (bool): discrete or continuous action space
            pretrain (str): ./data/local/{env}/{pretrain}/, the path to load pretrained cnn weights
            seed (int): Used to seed the random number generator to produce
                determinism.
            gpu (int): which gpu to use

        """
        deterministic.set_seed(seed)
        runner = LocalRunner(snapshot_config=ctxt)

        # make the env, given name and whether to use image obs
        env_name = env
        env = make_env(env_name, image, discrete)

        # make cnn encoder if learning from images
        cnn_encoder = None
        obs_dim = env.spec.observation_space.flat_dim
        if image:
            print('Using IMAGE observations!')
            cnn_encoder = CNNEncoder(in_channels=1,
                                        output_dim=256)
            # optionally load pre-trained weights
            if pretrain:
                print('Loading pre-trained weights from {}...'.format(pretrain))
                path_to_weights = f'data/local/{env_name}/{pretrain}/encoder.pth'
                cnn_encoder.load_state_dict(torch.load(path_to_weights))
                print('Success!')
            obs_dim = cnn_encoder.output_dim

        # discrete or continuous action space
        if discrete:
            q_function = DiscreteMLPQFunction
            input_action = False
            algo = DiscreteSAC
            policy = CategoricalMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=[256, 256],
                hidden_nonlinearity=nn.ReLU,
            )
            qf_input_dim = obs_dim

        else:
            q_function = ContinuousMLPQFunction
            input_action = True
            algo = SAC
            policy = TanhGaussianMLPPolicy(
                env_spec=env.spec,
                hidden_sizes=[256, 256],
                hidden_nonlinearity=nn.ReLU,
                output_nonlinearity=None,
                min_std=np.exp(-20.),
                max_std=np.exp(2.),
                cnn_encoder=cnn_encoder,
            )
            qf_input_dim = obs_dim + env.spec.action_space.flat_dim

        # make Q functions, rb, and algo
        qf1_mlp = q_function(env_spec=env.spec,
                                    input_dim=qf_input_dim,
                                    hidden_sizes=[256, 256],
                                    hidden_nonlinearity=F.relu)
        qf1 = CompositeQFunction(cnn_encoder, qf1_mlp, input_action=input_action)

        qf2_mlp = q_function(env_spec=env.spec,
                                    input_dim=qf_input_dim,
                                    hidden_sizes=[256, 256],
                                    hidden_nonlinearity=F.relu)
        qf2 = CompositeQFunction(cnn_encoder, qf2_mlp, input_action=input_action)

        replay_buffer = PathBuffer(capacity_in_transitions=int(1e6))

        sac = algo(env_spec=env.spec,
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
                cnn_encoder=cnn_encoder,
                train_cnn=train_cnn)

        if torch.cuda.is_available():
            set_gpu_mode(True, gpu_id=gpu)
        else:
            set_gpu_mode(False)
        sac.to()
        runner.setup(algo=sac, env=env, sampler_cls=LocalSampler)
        runner.train(n_epochs=1000, batch_size=1000)

    sac_batch(env=env, image=image, discrete=discrete, seed=seed, gpu=gpu, pretrain=pretrain, train_cnn=train_cnn)

main()
