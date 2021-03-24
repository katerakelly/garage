from garage.np.algos import RLAlgorithm
from garage.torch import dict_np_to_torch, global_device


class DataCollector(RLAlgorithm):
    """
    Class that executes a policy and saves the data
    """

    def __init__(self, policy, replay_buffer, steps_per_epoch, max_path_length, min_buffer_size=int(1e4), image=False):
        self.policy = policy
        self.replay_buffer = replay_buffer
        self._steps_per_epoch = steps_per_epoch
        self.max_path_length = max_path_length
        self._min_buffer_size = min_buffer_size
        self._image = image

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
                batch_size = int(self._min_buffer_size)
                runner.step_path = runner.obtain_samples(
                    runner.step_itr, batch_size)
                for path in runner.step_path:
                    d = dict(observation=path['observations'],
                             action=path['actions'],
                             reward=path['rewards'].reshape(-1, 1),
                             next_observation=path['next_observations'],
                             terminal=path['dones'].reshape(-1, 1))
                    if self._image:
                        # garage doesn't handle dicts well right now
                        d['env_info'] = path['env_infos']['state']
                    self.replay_buffer.add_path(d)

            runner.step_itr += 1

    def to(self, device=None):
        """Put all the networks within the model on device.

        Args:
            device (str): ID of GPU or CPU.

        """
        if device is None:
            device = global_device()
            self.policy.to(device)
