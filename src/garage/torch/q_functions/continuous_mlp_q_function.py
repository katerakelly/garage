"""This modules creates a continuous Q-function network."""

import torch
from torch import nn

from garage.torch.modules import MLPModule


class ContinuousMLPQFunction(MLPModule):
    """
    Implements a continuous MLP Q-value network.

    It predicts the Q-value for all actions based on the input state. It uses
    a PyTorch neural network module to fit the function of Q(s, a).
    """

    def __init__(self, env_spec, input_dim, **kwargs):
        """
        Initialize class with multiple attributes.

        Args:
            env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
            nn_module (nn.Module): Neural network module in PyTorch.
        """
        self._env_spec = env_spec
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim

        MLPModule.__init__(self,
                           input_dim=input_dim,
                           output_dim=1,
                           **kwargs)

    def forward(self, observations, actions):
        """Return Q-value(s)."""
        _input = torch.cat([observations, actions], 1)
        return super().forward(_input)


class ContinuousCNNQFunction(nn.Module):
    """
    Q-function network with optional CNN encoder on the front.
    """

    def __init__(self, cnn_encoder, mlp_q):
        super().__init__()
        self._cnn_encoder = cnn_encoder
        self._mlp_q = mlp_q

    def forward(self, observations, actions):
        if self._cnn_encoder is not None:
            observations = self._cnn_encoder(observations)
        return self._mlp_q(observations, actions)

