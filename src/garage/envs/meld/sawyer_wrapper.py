import os
import gym
from gym import spaces
import numpy as np
from PIL import Image
from gym.envs.mujoco.mujoco_env import convert_observation_to_space

SCRIPT_DIR = os.path.dirname(__file__)


class MeldSawyerWrapper(gym.Wrapper):
    '''
    wrap MELD Sawyer-based environments for use with garage
    each child wrapper for specific env should overwrite
    - get_image()
    - self.num_train_tasks
    '''
    def __init__(self, env, image_obs=False, task=None):
        self.env = env
        self.image_obs = image_obs
        self.obs_len = len(self._get_obs())

        # training tasks
        self._task = task
        self.num_train_tasks = 15
        # NOTE this hack is needed for shelf env to instantiate correctly
        dummy_task = self.sample_tasks(1)[0]
        self.set_task(dummy_task)

        self.action_space = self.env.action_space
        action = self.action_space.sample()
        self.action_dim = len(action)

        self.reset()

        observation, _reward, done, _info = self.step(action)
        assert not done
        # don't rely on inner env for this b/c changes for pixels
        self._set_observation_space(observation)

    def _get_obs(self):
        ''' true underlying state '''
        return self.env.get_obs().astype(np.float32)

    def _get_image_obs(self):
        img = self.get_image().flatten().astype(np.float32) / 255.
        return img

    def step(self, action):
        # since env_infos is not returned in reset(), return the previous timesteps's features
        prev_state = self._get_obs()
        rl2obs = np.concatenate([prev_state, self.prev_action, [self.prev_reward], [self.prev_done]])
        aug_obs, reward, done, infos = self.env.step(action)
        obs = aug_obs[:self.obs_len] # discard rewards and other info
        self.prev_action = action
        self.prev_reward = reward
        self.prev_done = done
        infos = {'score': infos[0], 'task_name': str(self._task), 'state': rl2obs}
        if self.image_obs:
            obs = self._get_image_obs()
        return obs, reward, done, infos

    def reset(self):
        self.prev_action = np.zeros(self.action_dim)
        self.prev_reward = 0
        self.prev_done = 0
        if self.image_obs:
            _ = self.env.reset()
            return self._get_image_obs()
        else:
            aug_obs = self.env.reset()
            return aug_obs[:self.obs_len] # discard rewards and other info

    #### obs space, copied from gym cheetah env
    def _set_observation_space(self, observation):
        self.observation_space = convert_observation_to_space(observation)
        if self.image_obs: # need to change bounds
            low, high = 0.0, 1.0
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
        if is_eval or num_tasks < self.num_train_tasks:
            tasks = self.env.init_tasks(num_tasks, is_eval)
        else:
            # tasks can be list of general objects
            tasks = self.env.init_tasks(self.num_train_tasks, is_eval)
            # sample with replacement to get to total number of tasks needed
            indices = np.random.choice(range(self.num_train_tasks), num_tasks - self.num_train_tasks, replace=True)
            samples = [tasks[idx] for idx in indices]
            tasks = np.concatenate([samples, tasks]) # make sure every task is in there at least once
        return tasks

    def set_task(self, task):
        self._task = task
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
        self.__init__(env=self.env, task=state['task'])


class MeldReachingWrapper(MeldSawyerWrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_train_tasks = 60

    def get_image(self, width=64, height=64, camera_name='track'):
        # use sim.render to avoid MJViewer which doesn't seem to work without display
        is_vis = width >= 128
        if is_vis:
            self.env.goal_visibility(visible=True)

        # by rendering at half width, images will be center cropped
        # manual inspection determines this is ok for reacher
        ee_img = self.env.sim.render(
            width=width / 2,
            height=height,
            camera_name='track_aux_reach',
        )
        ee_img = np.flipud(ee_img)
        scene_img = self.env.sim.render(
            width=width / 2,
            height=height,
            camera_name='track',
        )
        scene_img = np.flipud(scene_img)
        img = np.concatenate([scene_img, ee_img], axis=1)
        assert img.shape == (width, height, 3)
        if is_vis:
            self.env.goal_visibility(visible=False)
        return img


class MeldButtonWrapper(MeldSawyerWrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_train_tasks = 15

    def get_image(self, width=64, height=64, camera_name='track'):
        # use sim.render to avoid MJViewer which doesn't seem to work without display
        is_vis = width >= 128
        if is_vis:
            self.env.goal_visibility(visible=True)

        # by rendering at half width, images will be center cropped
        # manual inspection determines this is ok for reacher
        ee_img = self.env.sim.render(
            width=width / 2,
            height=height,
            camera_name='track_aux_insert',
        )
        ee_img = np.flipud(ee_img)
        scene_img = self.env.sim.render(
            width=width / 2,
            height=height,
            camera_name='track',
        )
        scene_img = np.flipud(scene_img)
        img = np.concatenate([scene_img, ee_img], axis=1)
        assert img.shape == (width, height, 3)
        if is_vis:
            self.env.goal_visibility(visible=False)
        return img


class MeldPegWrapper(MeldSawyerWrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_train_tasks = 30

    def get_image(self, width=64, height=64, camera_name='track'):
        '''
        peg insertion uses two cameras: scene, end-effector
        return one array with both images concatenated along axis 0
        '''
        # vis. determined by size of requested image - sketchy!!
        is_vis = width >= 128
        if is_vis:
            self.env.goal_visibility(visible=True)

        # use sim.render to avoid MJViewer which doesn't seem to work without display
        ee_img = self.env.sim.render(
            width=width,
            height=height,
            camera_name='track_aux_insert',
        )
        ee_img = np.flipud(ee_img)
        scene_img = self.env.sim.render(
            width=width,
            height=height,
            camera_name='track',
        )
        scene_img = np.flipud(scene_img)
        img = np.concatenate([scene_img, ee_img], axis=1)
        # resize image to be square
        img = Image.fromarray(img)
        img = img.resize((width, height))
        img = np.array(img)
        assert img.shape == (width, height, 3)
        if is_vis:
            self.env.goal_visibility(visible=False)
        return img


class MeldShelfWrapper(MeldSawyerWrapper):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_train_tasks = 40

    def get_image(self, width=64, height=64, camera_name='track'):
        # use sim.render to avoid MJViewer which doesn't seem to work without display
        is_vis = width >= 128
        if is_vis:
            self.goal_visibility(visible=True)

        # render both images at full width, then downsample to make square
        ee_img = self.sim.render(
            width=width,
            height=height,
            camera_name='track_aux_shelf2',
        )
        ee_img = np.flipud(ee_img)
        scene_img = self.sim.render(
            width=width,
            height=height,
            camera_name='track',
        )
        scene_img = np.flipud(scene_img)
        img = np.concatenate([scene_img, ee_img], axis=1)
        img = Image.fromarray(img)
        img = img.resize((width, height))
        img = np.array(img)
        assert img.shape == (width, height, 3)
        if is_vis:
            self.goal_visibility(visible=False)
        return img

    def sample_tasks(self, num_tasks, is_eval=False):
        # NOTE loading pre-generated tasks here
        all_tasks = np.load(os.path.join(SCRIPT_DIR, 'shelf', 'train40_eval10tasks.npy'))
        # NOTE for train and test tasks to not overlap, must be 40 and 10
        if is_eval:
            tasks = all_tasks[-num_tasks:]
        else:
            if self.num_train_tasks > len(all_tasks):
                print('Not enough tasks available. Must re-generate xml files.')
                raise Exception
            train_tasks = all_tasks[:self.num_train_tasks]
            if num_tasks <= self.num_train_tasks:
                tasks = train_tasks[:num_tasks]
            else:
                indices = np.random.choice(range(self.num_train_tasks), num_tasks - self.num_train_tasks, replace=True)
                samples = [train_tasks[idx] for idx in indices]
                tasks = np.concatenate([samples, train_tasks]) # make sure every task is in there at least once
        self.env.assign_tasks(tasks)
        return tasks
