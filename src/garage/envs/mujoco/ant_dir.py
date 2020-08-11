import numpy as np

from garage.envs.mujoco.ant_multitask_base import MultitaskAntEnv


class AntDirEnv(MultitaskAntEnv):

    def __init__(self, task={}, n_tasks=2, forward_backward=True, randomize_tasks=True, **kwargs):
        self.forward_backward = forward_backward
        task = task or {'goal': 0.0}
        super(AntDirEnv, self).__init__(task, n_tasks, **kwargs)

    def step(self, action):
        torso_xyz_before = np.array(self.get_body_com("torso"))

        direct = (np.cos(self._goal), np.sin(self._goal))

        self.do_simulation(action, self.frame_skip)
        torso_xyz_after = np.array(self.get_body_com("torso"))
        torso_velocity = torso_xyz_after - torso_xyz_before
        forward_reward = np.dot((torso_velocity[:2]/self.dt), direct)

        ctrl_cost = .5 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
                  and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
            torso_velocity=torso_velocity,
        )

    def sample_tasks(self, num_tasks):
        if self.forward_backward:
            #assert num_tasks == 2
            velocities = np.array([0., np.pi])
            if num_tasks == 1:
                velocities = np.array([np.random.choice(velocities)])
                print('Sampling 1 task: {}'.format(velocities[0]))
        else:
            velocities = np.random.uniform(0., 2.0 * np.pi, size=(num_tasks,))
        tasks = [{'goal': velocity} for velocity in velocities]
        return tasks

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
