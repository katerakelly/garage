import torch

from garage.torch.modules import MLPModule



class DiscreteMLPQFunction(MLPModule):
    """
    Implements a Q-value network for discrete-action MDPs

    Predict the Q-value for all actions given the input state,
    using a fully connected network to model the Q-function.
    """

    def __init__(self, env_spec, **kwargs):
        self._env_spec = env_spec
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        MLPModule.__init__(self,
                           input_dim=self._obs_dim,
                           output_dim=self._action_dim,
                           **kwargs)

    def forward(self, observations):
        """Return Q-value(s)."""
        return super().forward(observations)
