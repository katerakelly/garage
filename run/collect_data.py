#!/usr/bin/env python3
""" Collect and save data from a random or trained policy """
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
from garage.np.policies import RandomPolicy, StaticPolicy
from garage.torch.algos import DataCollector
from garage.misc.exp_util import make_env, make_exp_name


@click.command()
@click.argument('config', default=None)
@click.option('--name', default=None)
@click.option('--gpu', default=0)
@click.option('--debug', is_flag=True)
@click.option('--overwrite', is_flag=True)
def main(config, name, gpu, debug, overwrite):
    with open(os.path.join(config)) as f:
        variant = json.load(f)
    variant['gpu'] = gpu
    name = make_exp_name(name, debug)
    name = f'data/{name}'
    if debug:
        overwrite = True # always allow overwriting on a debug exp
    @wrap_experiment(prefix=variant['env'], name=name, snapshot_mode='none', archive_launch_repo=False, use_existing_dir=overwrite)
    def collect_data(ctxt, variant):
        """Set up environment and algorithm and run the task.

        Args:
            ctxt (garage.experiment.ExperimentContext): The experiment
                configuration used by LocalRunner to create the snapshotter.
            seed (int): Used to seed the random number generator to produce
                determinism.

        """
        deterministic.set_seed(variant['seed'])
        runner = LocalRunner(snapshot_config=ctxt)

        # make the env, given name and whether to use image obs
        image = variant['image']
        env = make_env(variant['env'], image, bg=variant['bg'], discrete=variant['discrete'])
        action_dim = env.spec.action_space.flat_dim

        policy = variant['policy']
        if policy == 'random':
            policies = [RandomPolicy(env_spec=env.spec)]
        elif policy == 'static':
            static_policy = StaticPolicy(env_spec=env.spec, action=action_dim-1)
            policies = [static_policy]
        elif policy == 'random-static':
            random_policy = RandomPolicy(env_spec=env.spec)
            static_policy = StaticPolicy(env_spec=env.spec, action=action_dim-1)
            policies = [random_policy, static_policy]
        elif policy == 'random-gripper':
            if 'gripper' not in variant['env']:
                print('Cannot collect gripper data in non-gripper env')
                raise Exception
            random_policy = RandomPolicy(env_spec=env.spec)
            policies = [random_policy]
            for act in [4, 5]:
                policies.append(StaticPolicy(env_spec=env.spec, action=act))
        else:
            raise NotImplementedError

        # if collecting images, do not make this too large, or will get a memory error
        num_collect = int(5e4)
        replay_buffer = PathBuffer(capacity_in_transitions=num_collect)

        for policy in policies:
            print(f'Collecting from policy: {policy}')
            # set min buffer size to num_collect in order to collect in a single epoch
            algo = DataCollector(policy, replay_buffer, steps_per_epoch=1, max_path_length=500, min_buffer_size=num_collect // len(policies), image=image)

            runner.setup(algo=algo, env=env, sampler_cls=LocalSampler)
            # every epoch, and every step per epoch, max(batch_size, traj_length) samples will be collected
            # the first iteration, batch_size will be overridden by min_buffer_size specified in algo
            runner.train(n_epochs=1, batch_size=1)
            print('Number transitions:', replay_buffer.n_transitions_stored)

        # save the replay buffer
        runner.simple_save({'replay_buffer': replay_buffer}, name='replay_buffer')

    collect_data(variant=variant)

main()
