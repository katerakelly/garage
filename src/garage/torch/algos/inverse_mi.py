import torch
import torch.nn.functional as F
from torch import nn

from garage.torch.algos import ULAlgorithm


class InverseMI(ULAlgorithm):
    """
    UL algorithm that trains an inverse model p(a | o_t, o_t+1)
    """
    def __init__(self, predictor, replay_buffer, discrete=True, **kwargs):
        super().__init__(predictor, replay_buffer, **kwargs)
        self._discrete = discrete
        if self._discrete:
            self._loss = nn.CrossEntropyLoss()
        else:
            self._loss = F.mse_loss

    def optimize_predictor(self, samples_data):
        obs = samples_data['observation']
        next_obs = samples_data['next_observation']
        actions = samples_data['action']

        # compute the loss
        pred_actions = self.predictor([obs, next_obs])
        if self._discrete:
            if actions.shape[-1] == 1:
                print('Cannot compute cross entropy loss with continuous action targets')
                raise Exception
            loss = self._loss(pred_actions, torch.argmax(actions, dim=-1))
        else:
            if len(pred_actions.flatten()) != len(actions.flatten()):
                print('Predicted and target actions do not have same shape. Are you using continuous target actions')
                raise Exception
            loss = self._loss(pred_actions.flatten(), actions.flatten())

        # optimize the predictor
        self._predictor_optimizer.zero_grad()
        loss.backward()
        self._predictor_optimizer.step()

        return loss

