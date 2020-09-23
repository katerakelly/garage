import numpy as np
from garage.np.policies import Policy

class OneHotPolicy(Policy):

    def __init__(self, env_spec):
        self._env_spec = env_spec
        self._action_dim = env_spec.action_space.flat_dim

    def get_action(self, observation):
        action, info = self.sample_action(observation)
        one_hot_action = self._index_to_one_hot(action).squeeze(0)
        return one_hot_action, info

    def get_actions(self, observations):
        return [self._index_to_one_hot(self.get_action(ob).squeeze(0)) for ob in observations], {}

    def get_param_values(self):
        """Return policy params (there are none).

        Returns:
            tuple: Empty tuple.

        """
        # pylint: disable=no-self-use
        return ()

    def set_param_values(self, params):
        """Set param values of policy.

        Args:
            params (object): Ignored.

        """
        # pylint: disable=no-self-use
        del params

    def _index_to_one_hot(self, arr):
        # turn actions into 1-hot vectors of length action_dim
        # to respect replay buffer interface
        if type(arr) is int:
            arr = np.array([arr])
        one_hot = np.zeros((arr.size, self._action_dim))
        one_hot[np.arange(arr.size), arr] = 1
        return one_hot


class RandomPolicy(OneHotPolicy):

    def sample_action(self, observation):
        return self._env_spec.action_space.sample(), {}


class StaticPolicy(OneHotPolicy):

    def sample_action(self, observation):
        # TODO hard-coded for catcher env to return none action
        return 2, {}
