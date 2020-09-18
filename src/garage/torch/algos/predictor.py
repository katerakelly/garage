import abc
import torch
import torch.nn.functional as F
from torch import nn


class Predictor(abc.ABC, nn.Module):
    """
    Base class for prediction module that consists of CNN encoder
    and head modules
    """
    def __init__(self,
                 cnn_encoder,
                 head):
        super().__init__()
        self.cnn_encoder = cnn_encoder
        self.head = head

    def forward(self, inputs):
        """ pass inputs through cnn encoder (if it exists) then head """
        if self.cnn_encoder is not None:
            # TODO this might be slower than reshaping images into batch dim
            feats = [self.cnn_encoder(in_) for in_ in inputs]
        else:
            feats = inputs
        feat = torch.cat(feats, dim=-1)
        return self.head(feat)

    def compute_loss(self, samples_data):
        """ compute the prediction loss """

    def evaluate(self, samples_data):
        """ evaluate predictor with appropriate metrics """


class InverseMI(Predictor):
    """
    UL algorithm that trains an inverse model p(a | o_t, o_t+1)
    """
    def __init__(self, cnn_encoder, head, discrete=True):
        super().__init__(cnn_encoder, head)
        self._discrete = discrete
        if self._discrete:
            self._loss = nn.CrossEntropyLoss()
        else:
            self._loss = F.mse_loss

    def compute_loss(self, samples_data):
        obs = samples_data['observation']
        next_obs = samples_data['next_observation']
        actions = samples_data['action']

        # compute the loss
        pred_actions = self.forward([obs, next_obs])
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
        return loss

    def evaluate(self, samples_data):
        """ return dict of (stat name, value) """
        obs = samples_data['observation']
        next_obs = samples_data['next_observation']
        actions = samples_data['action']

        if not self._discrete:
            raise NotImplementedError
        pred_actions = torch.argmax(self.forward([obs, next_obs]), dim=-1)
        actions = torch.argmax(actions, dim=-1)
        correct = (actions == pred_actions).sum().item()
        return {'accuracy': correct / len(actions)}


class Regressor(Predictor):
    """
    Predictor that regresses to a target with MSE loss
    given the current observation
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def _data_key(self):
        raise NotImplementedError

    def compute_loss(self, samples_data):
        obs = samples_data['observation']
        target = samples_data[self._data_key]

        # compute the loss
        pred = self.forward([obs])
        loss = F.mse_loss(pred.flatten(), target.flatten())

        return loss

    def evaluate(self, samples_data):
        """ report mean squared error """
        obs = samples_data['observation']
        state = samples_data[self._data_key]

        pred_state = self.forward([obs])
        mse = F.mse_loss(pred_state.flatten(), state.flatten())
        return {'MSE': mse.item()}


class RewardDecoder(Regressor):
    """
    Predict the reward given the current observation
    """
    @property
    def _data_key(self):
        return 'reward'


class StateDecoder(Regressor):
    """
    Predict ground truth state from current observation
    """
    @property
    def _data_key(self):
        return 'env_info'
