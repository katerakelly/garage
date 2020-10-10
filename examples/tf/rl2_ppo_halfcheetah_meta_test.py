#!/usr/bin/env python3
"""Example script to run RL2PPO meta test in HalfCheetah."""
# pylint: disable=no-value-for-parameter
import click

from garage import wrap_experiment
from garage.envs.garage_env import GarageEnv
from garage.envs.meld import HalfCheetahVelEnv as MeldHalfCheetahVelEnv
from garage.envs.meld import SawyerReachingEnvMultitask, SawyerPegInsertionEnv4Box, SawyerPegShelfEnvMultitask, SawyerButtonsEnv
from garage.envs.meld import MeldCheetahWrapper, MeldReachingWrapper, MeldPegWrapper, MeldShelfWrapper, MeldButtonWrapper
from garage.experiment import LocalTFRunner
from garage.experiment import task_sampler
from garage.experiment.experiment import ExperimentContext
from garage.experiment.deterministic import set_seed
from garage.experiment.meta_evaluator import MetaEvaluator
from garage.np.baselines import LinearFeatureBaseline
from garage.sampler import LocalSampler
from garage.tf.algos import RL2PPO
from garage.tf.algos.rl2 import RL2Env
from garage.tf.algos.rl2 import RL2Worker
from garage.tf.policies import GaussianGRUPolicy

## MELD settings
# cheetah: path_length: 50, train_tasks: 20, eval_tasks: 10
# reacher: path_length: 40, train_tasks: 60, eval_tasks: 10
# peg: path_length: 40, train_tasks: 30, eval_tasks: 10
# shelf: path_length: 40, train_tasks: 40, eval_tasks: 10
# button 2 traj/trial: path_legnth: 40, train_tasks: 15, eval_tasks: 12

@click.command()
@click.option('--seed', default=1)
@click.option('--max_path_length', default=40)
@click.option('--meta_batch_size', default=40) # change to 200 from state
@click.option('--n_epochs', default=500)
@click.option('--episode_per_task', default=2)
@click.option('--num_eval_exp_traj', default=1)
@click.option('--num_eval_test_traj', default=1)
@click.option('--env', default='cheetah')
@click.option('--gpu', default=0)
@wrap_experiment(prefix='rl2-ppo-image', archive_launch_repo=False)
def rl2_ppo_halfcheetah_meta_test(ctxt, seed, max_path_length, meta_batch_size,
        n_epochs, episode_per_task, num_eval_exp_traj, num_eval_test_traj, env, gpu):
    """Perform meta-testing on RL2PPO with HalfCheetah environment.

    Args:
        ctxt (garage.experiment.ExperimentContext): The experiment
            configuration used by LocalRunner to create the snapshotter.
        seed (int): Used to seed the random number generator to produce
            determinism.
        max_path_length (int): Maximum length of a single rollout.
        meta_batch_size (int): Meta batch size.
        n_epochs (int): Total number of epochs for training.
        episode_per_task (int): Number of training episode per task.

    """
    set_seed(seed)
    ctxt = ExperimentContext(snapshot_dir='', snapshot_mode='none', snapshot_gap='')
    with LocalTFRunner(snapshot_config=ctxt, gpu=gpu) as runner:
        # handle pixel normalization ourselves in the env!
        if env == 'cheetah':
            env = GarageEnv(MeldCheetahWrapper(MeldHalfCheetahVelEnv(), image_obs=True), is_image=False)
        elif env == 'reacher':
            env = GarageEnv(MeldReachingWrapper(SawyerReachingEnvMultitask(), image_obs=True), is_image=False)
        elif env == 'peg':
            env = GarageEnv(MeldPegWrapper(SawyerPegInsertionEnv4Box(), image_obs=True), is_image=False)
        elif env == 'shelf':
            env = GarageEnv(MeldShelfWrapper(SawyerPegShelfEnvMultitask(), image_obs=True), is_image=False)
        elif env == 'button':
            env = GarageEnv(MeldButtonWrapper(SawyerButtonsEnv(), image_obs=True), is_image=False)
        tasks = task_sampler.SetTaskSampler(lambda: RL2Env(
            env=env))
        test_tasks = task_sampler.SetTaskSampler(lambda: RL2Env(
            env=env), is_eval=True)
        env_spec = RL2Env(env=env).spec
        policy = GaussianGRUPolicy(name='policy',
                                   hidden_dim=64,
                                   env_spec=env_spec,
                                   state_include_action=False)

        baseline = LinearFeatureBaseline(env_spec=env_spec)

        meta_evaluator = MetaEvaluator(test_task_sampler=test_tasks,
                                       n_exploration_traj=num_eval_exp_traj,
                                       n_test_rollouts=num_eval_test_traj,
                                       max_path_length=max_path_length,
                                       n_test_tasks=12)

        algo = RL2PPO(rl2_max_path_length=max_path_length,
                      meta_batch_size=meta_batch_size,
                      task_sampler=tasks,
                      env_spec=env_spec,
                      policy=policy,
                      baseline=baseline,
                      discount=0.99,
                      gae_lambda=1.0,
                      lr_clip_range=0.2,
                      optimizer_args=dict(
                          batch_size=max_path_length * episode_per_task * meta_batch_size,
                          max_epochs=5,
                      ),
                      stop_entropy_gradient=True,
                      entropy_method='no_entropy',
                      policy_ent_coeff=0,
                      center_adv=True,
                      max_path_length=max_path_length * episode_per_task,
                      meta_evaluator=meta_evaluator,
                      n_epochs_per_eval=10)

        runner.setup(algo,
                     tasks.sample(meta_batch_size),
                     sampler_cls=LocalSampler,
                     n_workers=meta_batch_size,
                     worker_class=RL2Worker,
                     worker_args=dict(n_paths_per_trial=episode_per_task))

        runner.train(n_epochs=n_epochs,
                     batch_size=episode_per_task * max_path_length *
                     meta_batch_size)


rl2_ppo_halfcheetah_meta_test()
