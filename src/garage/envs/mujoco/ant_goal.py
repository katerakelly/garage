import numpy as np

from garage.envs.mujoco.ant_multitask_base import MultitaskAntEnv


# Copy task structure from https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_goal.py
class AntGoalEnv(MultitaskAntEnv):
    def __init__(self, task={}, n_tasks=2, **kwargs):
        task = task or {'goal': np.array([0, 0])}
        super(AntGoalEnv, self).__init__(task, n_tasks, **kwargs)

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))

        goal_reward = -np.sum(np.abs(xposafter[:2] - self._goal)) # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
        )

    def sample_tasks(self, num_tasks):
        # hard-code a single task for debugging
        if num_tasks == 1:
            return [{'goal': np.array([1.0, 1.0])}]
        a = np.random.random(num_tasks) * 2 * np.pi
        r = 3 * np.random.random(num_tasks) ** 0.5
        goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

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
