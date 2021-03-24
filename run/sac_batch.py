#!/usr/bin/env python3
"""This is an example to train a task with SAC algorithm written in PyTorch."""
import os
import json
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
@click.argument('config', default=None)
@click.option('--name', default=None)
@click.option('--gpu', default=0)
@click.option('--seed', default=1)
@click.option('--debug', is_flag=True)
@click.option('--overwrite', is_flag=True)
@click.option('--snapshot', is_flag=True)
def main(config, name, gpu, seed, debug, overwrite, snapshot):
    with open(os.path.join(config)) as f:
        variant = json.load(f)
    variant['gpu'] = gpu
    variant['seed'] = seed
    name = make_exp_name(name, debug)
    name = f'rl/{name}'
    if debug:
        overwrite = True # always allow overwriting on a debug exp
    snapshot_mode = 'last' if snapshot else 'none'
    @wrap_experiment(prefix=variant['env'], name=name, snapshot_mode=snapshot_mode, archive_launch_repo=False, use_existing_dir=overwrite)
    def sac_batch(ctxt, variant):
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
        # unpack commonly used args
        image = variant['image']
        discrete = variant['discrete']

        print('Setting seed = {}'.format(variant['seed']))
        deterministic.set_seed(variant['seed'])
        runner = LocalRunner(snapshot_config=ctxt)

        # make the env, given name and whether to use image obs
        env_name = variant['env']
        env = make_env(env_name, image, variant['bg'], discrete)

        # make cnn encoder if learning from images
        cnn_encoder = None
        obs_dim = env.spec.observation_space.flat_dim
        if image:
            print('Using IMAGE observations!')
            cnn_encoder = CNNEncoder(in_channels=1,
                                        output_dim=256)
            # optionally load pre-trained weights
            pretrain = variant['pretrain']
            if pretrain:
                print('Loading pre-trained weights from {}...'.format(pretrain))
                path_to_weights = f'output/{env_name}/ul/{pretrain}/encoder.pth'
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
                train_cnn=variant['train_cnn'])

        if torch.cuda.is_available():
            set_gpu_mode(True, gpu_id=variant['gpu'])
        else:
            set_gpu_mode(False)
        sac.to()
        runner.setup(algo=sac, env=env, sampler_cls=LocalSampler)
        runner.train(n_epochs=1000, batch_size=1000)

    sac_batch(variant=variant)

main()
