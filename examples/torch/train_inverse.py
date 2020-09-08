#!/usr/bin/env python3
"""This is an example to train a task with SAC algorithm written in PyTorch."""
import gym
import numpy as np
import click
import datetime
import dateutil.tz
import pickle as pkl
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
from garage.torch.modules import CNNEncoder
from garage.torch.algos import InverseMI
from garage.torch.modules import GaussianMLPTwoHeadedModule

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

    @wrap_experiment(prefix=env, name=name, snapshot_mode='none', archive_launch_repo=False)
    def train_inverse(ctxt, env, image, seed, gpu):
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
        action_dim = env.spec.action_space.flat_dim
        obs_dim = env.spec.observation_space.flat_dim

        if image:
            raise NotImplementedError
            # TODO need to put both images through conv!
            print('Using IMAGE observations!')
            cnn_encoder = CNNEncoder(in_channels=4,
                                        output_dim=256)
            input_dim = cnn_encoder.output_dim * 2

        # make mlp to predict actions
        mlp_encoder = GaussianMLPTwoHeadedModule(input_dim=obs_dim * 2,
                                                 output_dim=action_dim,
                                                 hidden_sizes=[256, 256],
                                                 hidden_nonlinearity=nn.ReLU,
                                                 min_std=np.exp(-20.),
                                                 max_std=np.exp(2.))
        if image:
            predictor = nn.Sequential(cnn_encoder, mlp_encoder)
        else:
            predictor = mlp_encoder

        # load saved rb
        rb_filename = '/home/rakelly/garage/data/local/cheetah/debug/replay_buffer.pkl'
        with open(rb_filename, 'rb') as f:
            replay_buffer = pkl.load(f)['replay_buffer']
        f.close()
        # test the obs to make sure it is what we expect
        test_obs = replay_buffer.sample_transitions(batch_size=1)['observation']
        assert len(test_obs.flatten()) == env.spec.observation_space.flat_dim

        # construct algo and train!
        algo = InverseMI(predictor,
                         replay_buffer,
                         buffer_batch_size=256)

        if torch.cuda.is_available():
            set_gpu_mode(True, gpu_id=gpu)
        else:
            set_gpu_mode(False)
        algo.to()
        runner.setup(algo=algo, env=env)
        runner.train(n_epochs=1000, batch_size=1000)

    train_inverse(env=env, image=image, seed=seed, gpu=gpu)

main()
