"""CategoricalMLPPolicy."""
import numpy as np
import torch
from torch import nn

from garage.torch import global_device
from garage.torch.modules import MLPModule
from garage.torch.policies.stochastic_policy import StochasticPolicy


class CategoricalMLPPolicy(StochasticPolicy):
    """MLP whose outputs are fed through a softmax and then become the probs
    in a categorical distribution

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        hidden_sizes (list[int]): Output dimension of dense layer(s) for
            the MLP for mean. For example, (32, 32) means the MLP consists
            of two hidden layers, each with 32 hidden units.
        hidden_nonlinearity (callable): Activation function for intermediate
            dense layer(s). It should return a torch.Tensor. Set it to
            None to maintain a linear activation.
        hidden_w_init (callable): Initializer function for the weight
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        hidden_b_init (callable): Initializer function for the bias
            of intermediate dense layer(s). The function should return a
            torch.Tensor.
        output_w_init (callable): Initializer function for the weight
            of output dense layer(s). The function should return a
            torch.Tensor.
        output_b_init (callable): Initializer function for the bias
            of output dense layer(s). The function should return a
            torch.Tensor.
        layer_normalization (bool): Bool for using layer normalization or not.
        name (str): Name of policy.

    """

    def __init__(self,
                 env_spec,
                 hidden_sizes=(32, 32),
                 hidden_nonlinearity=torch.tanh,
                 hidden_w_init=nn.init.xavier_uniform_,
                 hidden_b_init=nn.init.zeros_,
                 output_nonlinearity=None,
                 output_w_init=nn.init.xavier_uniform_,
                 output_b_init=nn.init.zeros_,
                 layer_normalization=False,
                 name='CategoricalMLPPolicy'):
        super().__init__(env_spec, name)
        self._obs_dim = env_spec.observation_space.flat_dim
        self._action_dim = env_spec.action_space.flat_dim
        self._module = MLPModule(
            input_dim=self._obs_dim,
            output_dim=self._action_dim,
            hidden_sizes=hidden_sizes,
            hidden_nonlinearity=hidden_nonlinearity,
            hidden_w_init=hidden_w_init,
            hidden_b_init=hidden_b_init,
            output_nonlinearity=None,
            output_w_init=output_w_init,
            output_b_init=output_b_init,
            layer_normalization=layer_normalization)

    def forward(self, observations):
        """Compute the action distributions from the observations.

        Args:
            observations (torch.Tensor): Batch of observations on default
                torch device.

        Returns:
            torch.distributions.Distribution: Batch distribution of actions.
            dict[str, torch.Tensor]: Additional agent_info, as torch Tensors

        """
        probs = self._module(observations)
        probs = torch.softmax(probs, dim=-1)
        return probs, {}

    def _index_to_one_hot(self, arr):
        # turn actions into 1-hot vectors of length action_dim
        # to respect replay buffer interface
        one_hot = np.zeros((arr.size, self._action_dim))
        one_hot[np.arange(arr.size), arr] = 1
        return one_hot

    def get_action(self, observation):
        with torch.no_grad():
            if not isinstance(observation, torch.Tensor):
                observation = torch.as_tensor(observation).float().to(
                    global_device())
            observation = observation.unsqueeze(0)
            probs, info = self.forward(observation)
            dist = torch.distributions.Categorical(probs=probs)
            action, info = dist.sample().squeeze(0).cpu().numpy(), {
                k: v.squeeze(0).detach().cpu().numpy()
                for (k, v) in info.items()
            }
        one_hot_action = self._index_to_one_hot(action).squeeze(0)
        return one_hot_action, info

    def get_actions(self, observations):
        with torch.no_grad():
            if not isinstance(observations, torch.Tensor):
                observations = torch.as_tensor(observations).float().to(
                    global_device())
            probs, info = self.forward(observations)
            dist = torch.distributions.Categorical(probs=probs)
            actions, info = dist.sample().cpu().numpy(), {
                k: v.detach().cpu().numpy()
                for (k, v) in info.items()
            }
        one_hot_actions = self._index_to_one_hot(actions)
        return one_hot_actions, info
