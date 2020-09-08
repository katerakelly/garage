from garage.np.algos import RLAlgorithm


class DataCollector(RLAlgorithm):
    """
    Class that executes a policy and saves the data
    """

    def __init__(self, policy, replay_buffer, steps_per_epoch, max_path_length, min_buffer_size=int(1e4)):
        self.policy = policy
        self.replay_buffer = replay_buffer
        self._steps_per_epoch = steps_per_epoch
        self.max_path_length = max_path_length
        self._min_buffer_size = min_buffer_size

    def train(self, runner):
        """
        Collect data with policy and add to replay buffer

        Args:
            runner (LocalRunner): LocalRunner is passed to give algorithm
                the access to runner.step_epochs(), which provides services
                such as snapshotting and sampler control.

        """
        for _ in runner.step_epochs():
            for _ in range(self._steps_per_epoch):
                if not (self.replay_buffer.n_transitions_stored >=
                        self._min_buffer_size):
                    batch_size = int(self._min_buffer_size)
                else:
                    batch_size = None
                runner.step_path = runner.obtain_samples(
                    runner.step_itr, batch_size)
                for path in runner.step_path:
                    self.replay_buffer.add_path(
                        dict(observation=path['observations'],
                             action=path['actions'],
                             reward=path['rewards'].reshape(-1, 1),
                             next_observation=path['next_observations'],
                             terminal=path['dones'].reshape(-1, 1)))
            runner.step_itr += 1
        # save the replay buffer
        runner.simple_save(runner.step_itr, {'replay_buffer': self.replay_buffer}, name='replay_buffer')

