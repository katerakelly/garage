#!/usr/bin/env python3
"""This is an example to train a task with SAC algorithm written in PyTorch."""
import os
import json
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
from garage.torch.algos import InverseMI, ULAlgorithm, CPC, RewardDecoder, ForwardMI, StateDecoder, Bisimulation, CloneQ
from garage.torch.modules import GaussianMLPTwoHeadedModule, MLPModule
from garage.torch.q_functions import ContinuousMLPQFunction, CompositeQFunction, DiscreteMLPQFunction
from garage.misc.exp_util import make_env, make_exp_name


@click.command()
@click.argument('config', default=None)
@click.option('--name', default=None)
@click.option('--gpu', default=0)
@click.option('--seed', default=1)
@click.option('--debug', is_flag=True)
@click.option('--overwrite', is_flag=True)
def main(config, name, gpu, seed, debug, overwrite):
    with open(os.path.join(config)) as f:
        variant = json.load(f)
    variant['gpu'] = gpu
    variant['seed'] = seed
    name = make_exp_name(name, debug)
    name = f'ul/{name}'
    if debug:
        overwrite = True # always allow overwriting on a debug exp
    @wrap_experiment(prefix=variant['env'], name=name, snapshot_mode='none', archive_launch_repo=False, use_existing_dir=overwrite)
    def train_inverse(ctxt, variant):
        """Set up environment and algorithm and run the task.

        Args:
            ctxt (garage.experiment.ExperimentContext): The experiment
                configuration used by LocalRunner to create the snapshotter.
            seed (int): Used to seed the random number generator to produce
                determinism.

        """
        # unpack commonly used args
        image = variant['image']
        discrete = variant['discrete']
        algo = variant['algo']

        print('Setting seed = {}'.format(variant['seed']))
        deterministic.set_seed(variant['seed'])
        runner = LocalRunner(snapshot_config=ctxt)

        # make the env, given name and whether to use image obs
        env_name = variant['env']
        bg = variant['bg'] if 'bg' in variant else None
        env = make_env(env_name, image, bg=bg, discrete=discrete)

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
            # optionally load pre-trained weights
            pretrain = variant['pretrain']
            if pretrain:
                print('Loading pre-trained weights from {}...'.format(pretrain))
                path_to_weights = f'output/{env_name}/ul/{pretrain}/encoder.pth'
                cnn_encoder.load_state_dict(torch.load(path_to_weights))
                print('Success!')
            hidden_sizes = [] # linear decoder from conv features -> force information into the conv encoder

        loss_weights = None
        momentum = 0.9
        snapshot_metric = None
        if 'predictQ' in algo:
            # instantiate proxy for optimal Q
            q_cnn_encoder = CNNEncoder(in_channels=1,
                                        output_dim=256)
            q_function = ContinuousMLPQFunction
            qf_input_dim = obs_dim + env.spec.action_space.flat_dim
            qf1_mlp = q_function(env_spec=env.spec,
                                        input_dim=qf_input_dim,
                                        hidden_sizes=[256, 256],
                                        hidden_nonlinearity=F.relu)
            qf1 = CompositeQFunction(q_cnn_encoder, qf1_mlp, input_action=True)
            qf2_mlp = q_function(env_spec=env.spec,
                                        input_dim=qf_input_dim,
                                        hidden_sizes=[256, 256],
                                        hidden_nonlinearity=F.relu)
            qf2 = CompositeQFunction(q_cnn_encoder, qf2_mlp, input_action=True)
            # load the pre-trained weights
            pretrain = variant['pretrain_Q']
            print('Loading pre-trained weights from {}...'.format(pretrain))
            path_to_weights_1 = f'output/{env_name}/rl/{pretrain}/critic_1.pth'
            qf1.load_state_dict(torch.load(path_to_weights_1))
            path_to_weights_2 = f'output/{env_name}/rl/{pretrain}/critic_2.pth'
            qf2.load_state_dict(torch.load(path_to_weights_2))
            print('Success!')

            # make predictor network with same architecture
            # use CNN with pre-trained UL weights
            q_predictor = ContinuousMLPQFunction
            qf_mlp_predictor = q_function(env_spec=env.spec,
                                        input_dim=qf_input_dim,
                                        hidden_sizes=[256, 256],
                                        hidden_nonlinearity=F.relu)
            qf_predictor = CompositeQFunction(cnn_encoder, qf_mlp_predictor, input_action=True)
            predictors = {'CloneQ': CloneQ(qf_predictor, [qf1, qf2], action_dim=action_dim, train_cnn=False)}

        elif 'inverse' in algo:
            # make mlp to predict actions
            action_mlp = MLPModule(input_dim=obs_dim * 2,
                                    output_dim=action_dim,
                                    hidden_sizes=hidden_sizes,
                                    hidden_nonlinearity=nn.ReLU)
            # pass inputs through CNN (if images), then though
            # mlp to predict actions
            if algo == 'inverse':
                predictors = {'InverseMI': InverseMI(cnn_encoder, action_mlp, action_dim=action_dim,  discrete=discrete, information_bottleneck=variant['ib'], kl_weight=variant['klw'])}
            elif algo == 'inverse-reward':
                reward_mlp = MLPModule(input_dim=obs_dim,
                                        output_dim=3,
                                        hidden_sizes=hidden_sizes,
                                        hidden_nonlinearity=nn.ReLU)
                predictors = {'InverseMI': InverseMI(cnn_encoder, action_mlp, action_dim=action_dim, discrete=discrete, information_bottleneck=variant['ib']), 'RewardDecode': RewardDecoder(cnn_encoder, reward_mlp, action_dim=action_dim)}
                loss_weights = {'InverseMI': 1.0, 'RewardDecode': 10.0}
        elif algo == 'cpc':
            predictors = {'CPC': CPC(cnn_encoder, action_dim=action_dim)}
            momentum = 0.0
            snapshot_metric = ['CPC', 'CELoss']
        elif algo == 'forward':
            predictors = {'ForwardMI': ForwardMI(cnn_encoder, action_dim=action_dim)}
            #momentum = 0.0
            snapshot_metric = ['ForwardMI', 'CELoss']
        elif algo == 'bisim':
            obs_dim = obs_dim // 2
            reward_mlp = MLPModule(input_dim=obs_dim,
                                   output_dim=1,
                                   hidden_sizes=hidden_sizes,
                                   hidden_nonlinearity=nn.ReLU)
            predictors = {'Bisimulation': Bisimulation(cnn_encoder, reward_mlp, action_dim, information_bottleneck=variant['ib'], kl_weight=variant['klw'])}
        elif algo == 'state-decode':
            output_dim = 5 if 'gripper' in env_name else 4
            state_mlp = MLPModule(input_dim=obs_dim,
                                  output_dim=output_dim,
                                  hidden_sizes=[256, 256],
                                  hidden_nonlinearity=nn.ReLU)
            gripper = 'gripper' in env_name
            predictors = {'StateDecode': StateDecoder(cnn_encoder, state_mlp, action_dim=action_dim, train_cnn=variant['train_cnn'])}
        else:
            print('Algorithm {} not implemented.'.format(algo))
            raise NotImplementedError

        # load saved rb
        rb = variant['rb']
        rb_filename = f'output/{env_name}/data/{rb}/replay_buffer.pkl'
        with open(rb_filename, 'rb') as f:
            replay_buffer = pkl.load(f)['replay_buffer']
        f.close()
        # test the obs to make sure it is what we expect
        test_obs = replay_buffer.sample_transitions(batch_size=1)['observation']
        assert len(test_obs.flatten()) == env.spec.observation_space.flat_dim

        # construct algo and train!
        algo = ULAlgorithm(predictors,
                           replay_buffer,
                           loss_weights = loss_weights,
                           lr=1e-2,
                           buffer_batch_size=256,
                           momentum=momentum,
                           snapshot_metric=snapshot_metric)

        if torch.cuda.is_available():
            set_gpu_mode(True, gpu_id=variant['gpu'])
        else:
            set_gpu_mode(False)
        algo.to()
        runner.setup(algo=algo, env=env)
        runner.train(n_epochs=1000, batch_size=1000)

    train_inverse(variant=variant)

main()
