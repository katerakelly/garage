from garage.np.policies import Policy

class RandomPolicy(Policy):

    def __init__(self, env_spec):
        self._env_spec = env_spec

    def get_action(self, observation):
        return self._env_spec.action_space.sample(), {}

    def get_actions(self, observations):
        return [self.get_action(ob) for ob in observations]

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
