import abc
from dowel import tabular
import torch

from garage.np.algos import RLAlgorithm
from garage.torch import dict_np_to_torch, global_device


class ULAlgorithm(RLAlgorithm, abc.ABC):
    """
    Base class for UL algorithms
    Supports multiple predictive models
    predictors: dict of (key, Predictor object)
    """
    def __init__(self,
                 predictors,
                 replay_buffer,
                 loss_weights=None,
                 lr=1e-3,
                 buffer_batch_size=64,
                 steps_per_epoch=100,
                 eval_batch_size=512):
        self.predictors = predictors
        self.replay_buffer = replay_buffer
        # if no loss weights given, weight equally
        self._loss_weights = loss_weights
        if loss_weights is None:
            self._loss_weights = dict([(k, 1.0) for k in predictors.keys()])
        self._lr = lr
        self._buffer_batch_size = buffer_batch_size
        self._steps_per_epoch = steps_per_epoch
        self._eval_batch_size = eval_batch_size

        params = []
        for p in self.predictors.values():
            params += list(p.parameters())
        self._optimizer = torch.optim.SGD(params, lr=self._lr)

    def train(self, runner):
        """ training loop """
        for _ in runner.step_epochs():
            for _ in range(self._steps_per_epoch):
                eval_dicts = self.train_once()
            self._log_statistics(eval_dicts)
            # NOTE: assumes predictors share conv encoder!!
            cnn_encoder = next(iter(self.predictors.values())).cnn_encoder
            if cnn_encoder is not None:
                runner.save_state_dict(cnn_encoder, 'encoder')
            runner.step_itr += 1

    def train_once(self):
        """ sample data and optimize predictors """
        samples = self.replay_buffer.sample_transitions(
            self._buffer_batch_size)
        samples = dict_np_to_torch(samples)
        losses, eval_dicts = {}, {}
        self._optimizer.zero_grad()
        for name, p in self.predictors.items():
            loss = p.compute_loss(samples) * self._loss_weights[name]
            loss.backward(retain_graph=True) # multiple predictors may optimize same network
            losses[name] = loss.item()
        self._optimizer.step()

        # perform evaluation of each predictor
        eval_samples = self.replay_buffer.sample_transitions(
            self._eval_batch_size)
        eval_samples = dict_np_to_torch(eval_samples)
        for name, p in self.predictors.items():
            eval_dicts[name] = p.evaluate(eval_samples)

        # add the loss to the eval_dict
        for k in eval_dicts.keys():
            eval_dicts[k].update({'loss': losses[k]})
        return eval_dicts

    def _log_statistics(self, eval_dict):
        for predictor_name, stat_dict in eval_dict.items():
            for k, v in stat_dict.items():
                tabular.record(f'{predictor_name}/{k}', v)

    @property
    def networks(self):
        return list(self.predictors.values())

    def to(self, device=None):
        if device is None:
            device = global_device()
        for net in self.networks:
            net.to(device)



