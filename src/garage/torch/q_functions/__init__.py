"""PyTorch Q-functions."""
from garage.torch.q_functions.continuous_mlp_q_function import (
    ContinuousMLPQFunction, ContinuousCNNQFunction)
from garage.torch.q_functions.discrete_mlp_q_function import (
    DiscreteMLPQFunction)

__all__ = ['ContinuousMLPQFunction', 'ContinuousCNNQFunction', 'DiscreteMLPQFunction']
