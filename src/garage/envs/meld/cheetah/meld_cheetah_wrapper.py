import gym
from gym import spaces
from gym.envs.mujoco.mujoco_env import convert_observation_to_space
import numpy as np

class MeldCheetahWrapper(gym.Wrapper):
    '''
    wrap a MELD env for use with garage
    '''
    def __init__(self, env, image_obs=False, task=None):
        self.env = env
        self.image_obs = image_obs
        self._task = task or {'velocity': 0.}
        self.obs_len = len(self._get_obs())

        # NOTE set action and obs spaces, copied from gym cheetah env
        self._set_action_space()
        action = self.action_space.sample()
        observation, _reward, done, _info = self.step(action)
        assert not done
        self._set_observation_space(observation)

    def _get_obs(self):
        if self.image_obs:
            return self.get_image().flatten()
        else:
            return self.env._get_obs().astype(np.float32)

    def step(self, action):
        aug_obs, reward, done, infos = self.env.step(action)
        obs = aug_obs[:self.obs_len] # discard rewards and other info
        infos = {'score': infos[0], 'task_name': str(self.env.target_vel)}
        if self.image_obs:
            obs = self._get_obs()
        return obs, reward, done, infos

    def reset(self):
        if self.image_obs:
            _ = self.env.reset()
            return self._get_obs()
        else:
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
        if self.image_obs: # need to change bounds
            low, high = 0, 255
            self.observation_space = spaces.Box(low=low, high=high, shape=self.observation_space.shape, dtype=np.uint8)
        return self.observation_space

    #### rendering settings
    def get_image(self, width=64, height=64, camera_name='track'):
        # use sim.render to avoid MJViewer which doesn't seem to work without display
        img = self.env.sim.render(
            width=width,
            height=height,
            camera_name=camera_name,
        )
        return np.flipud(img)

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
        self.__init__(env=self.env, task=state['task'])
