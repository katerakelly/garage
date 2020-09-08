import torch
import torch.nn.functional as F

from garage.torch.algos import ULAlgorithm


class InverseMI(ULAlgorithm):
    """
    UL algorithm that trains an inverse model p(a | o_t, o_t+1)
    """

    def optimize_predictor(self, samples_data):
        obs = samples_data['observation']
        next_obs = samples_data['next_observation']
        actions = samples_data['action']

        # compute the loss
        in_ = torch.cat([obs, next_obs], dim=-1)
        pred_actions = self.predictor(in_).mean
        loss = F.mse_loss(pred_actions.flatten(), actions.flatten())

        # optimize the predictor
        self._predictor_optimizer.zero_grad()
        loss.backward()
        self._predictor_optimizer.step()

        return loss

