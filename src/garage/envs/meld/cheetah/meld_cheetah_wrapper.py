import gym
from gym import spaces
from gym.envs.mujoco.mujoco_env import convert_observation_to_space
import numpy as np

class MeldCheetahWrapper(gym.Wrapper):
    '''
    wrap a MELD env for use with garage
    '''
    def __init__(self, env, task=None):
        self.env = env
        self._task = task or {'velocity': 0.}
        self.obs_len = len(self._get_obs())

        # NOTE set action and obs spaces, copied from gym cheetah env
        self._set_action_space()
        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done
        self._set_observation_space(observation)

    def _get_obs(self):
        return self.env._get_obs().astype(np.float32)

    def step(self, action):
        aug_obs, reward, done, infos = self.env.step(action)
        obs = aug_obs[:self.obs_len] # discard rewards and other info
        infos = {'score': infos[0], 'task_name': str(self.env.target_vel)}
        return obs, reward, done, infos

    def reset(self):
        aug_obs = self.env.reset()
        return aug_obs[:self.obs_len] # discard rewards and other info

    #### action and obs space, copied from gym cheetah env
    def _set_action_space(self):
        bounds = self.model.actuator_ctrlrange.copy().astype(np.float32)
        low, high = bounds.T
        self.action_space = spaces.Box(low=low, high=high, dtype=np.float32)
        return self.action_space

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    #### task-interface
    def sample_tasks(self, num_tasks, is_eval=False):
        num_train_tasks = 20 # match MELD
        if is_eval or num_tasks < num_train_tasks:
            velocities = self.env.init_tasks(num_tasks, is_eval)
        else:
            velocities = self.env.init_tasks(num_train_tasks, is_eval)
            # sample with replacement to get to total number of tasks needed
            samples = np.random.choice(velocities, num_tasks - num_train_tasks, replace=True)
            velocities = np.concatenate([samples, velocities]) # make sure every task is in there at least once
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks

    def set_task(self, task):
        self.env.set_task_for_env(task['velocity'])

    #### getters and setters copied from garage boilerplate
    def __getstate__(self):
        """See `Object.__getstate__.

        Returns:
            dict: The instance’s dictionary to be pickled.

        """
        return dict(task=self._task)

    def __setstate__(self, state):
        """See `Object.__setstate__.

        Args:
            state (dict): Unpickled state of this object.

        """
        self.__init__(task=state['task'])
