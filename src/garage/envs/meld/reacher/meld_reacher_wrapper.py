import gym
from gym import spaces
from gym.envs.mujoco.mujoco_env import convert_observation_to_space
import numpy as np

class MeldReacherWrapper(gym.Wrapper):
    '''
    wrap reacher env for use with garage
    '''
    def __init__(self, env, task=None):
        self.env = env
        self._task = task or {'velocity': 0.}
        self.obs_len = len(self._get_obs())

        self.action_space = self.env.action_space
        # re-compute obs space here w/o reward concat to obs
        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done
        self._set_observation_space(observation)

    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        return self.observation_space

    def _get_obs(self):
        return self.env.get_obs().astype(np.float32)

    def step(self, action):
        aug_obs, reward, done, infos = self.env.step(action)
        obs = aug_obs[:self.obs_len] # discard rewards and other info
        # convert infos from np array to dict format
        infos = {'score': infos[0], 'task_name': str(self.env.model.site_pos[self.env.site_id_goal])}
        return obs, reward, done, infos

    def reset(self):
        aug_obs = self.env.reset()
        return aug_obs[:self.obs_len] # discard rewards and other info

    #### task-interface
    def sample_tasks(self, num_tasks, is_eval=False):
        num_train_tasks = 60 # match MELD
        if is_eval or num_tasks < num_train_tasks:
            tasks = self.env.init_tasks(num_tasks, is_eval)
        else:
            goals = self.env.init_tasks(num_train_tasks, is_eval)
            # sample with replacement to get to total number of tasks needed
            samples = np.random.choice(list(range(len(goals))), num_tasks - num_train_tasks, replace=True)
            samples = [goals[s] for s in samples]
            tasks = list(samples) + list(goals) # make sure every task is in there at least once
        return tasks

    def set_task(self, task):
        self.env.set_task_for_env(task)

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
