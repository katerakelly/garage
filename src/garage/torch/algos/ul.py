import abc
from dowel import tabular
import torch

from garage.np.algos import RLAlgorithm
from garage.torch import dict_np_to_torch, global_device
from garage.torch.modules import CNNEncoder


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
                 eval_batch_size=512,
                 train_cnn=True):
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
        self._optimizer = torch.optim.SGD(params, lr=self._lr, momentum=0.9)

    def train(self, runner):
        """ training loop """
        dsize = self.replay_buffer._transitions_stored
        for _ in runner.step_epochs():
            # train
            for _ in range(self._steps_per_epoch):
                losses = self.train_once()
            #eval
            eval_dicts = self.eval_once()
            # add the loss to the eval_dict
            for k in eval_dicts.keys():
                eval_dicts[k].update(losses[k])
            # add the amount of data processed to logging
            eval_dicts['DataEpochs'] = ((runner.step_itr + 1) *self._steps_per_epoch * self._buffer_batch_size) / dsize
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
        stats = {}
        self._optimizer.zero_grad()
        for name, p in self.predictors.items():
            # losses and train stats should both be dicts
            loss_vals, train_stats = p.compute_loss(samples)
            for k, v in loss_vals.items():
                v.backward(retain_graph=True)
                d = {k: v.item()}
                if name not in stats:
                    stats[name] = d
                else:
                    stats[name].update(d)
            for k, v in train_stats.items():
                d = {k: v}
                if name not in stats:
                    stats[name] = d
                else:
                    stats[name].update(d)
        self._optimizer.step()
        return stats

    def eval_once(self):
        # perform evaluation of each predictor
        eval_samples = self.replay_buffer.sample_transitions(
            self._eval_batch_size)
        eval_samples = dict_np_to_torch(eval_samples)
        eval_dicts = {}
        for name, p in self.predictors.items():
            eval_dicts[name] = p.evaluate(eval_samples)
        return eval_dicts

    def _log_statistics(self, eval_dict):
        for key, value in eval_dict.items():
            if type(value) is dict:
                for k, v in value.items():
                    tabular.record(f'{key}/{k}', v)
            else:
                tabular.record(f'{key}', value)

    @property
    def networks(self):
        return list(self.predictors.values())

    def to(self, device=None):
        if device is None:
            device = global_device()
        for net in self.networks:
            net.to(device)



