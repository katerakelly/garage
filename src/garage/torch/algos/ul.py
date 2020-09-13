import abc
from dowel import tabular
import torch

from garage.np.algos import RLAlgorithm
from garage.torch import dict_np_to_torch, global_device


class ULAlgorithm(RLAlgorithm, abc.ABC):
    """
    Base class for UL algorithms
    """
    def __init__(self,
                 predictor,
                 replay_buffer,
                 lr=1e-3,
                 buffer_batch_size=64,
                 steps_per_epoch=100,
                 eval_batch_size=512):
        self.predictor = predictor
        self.replay_buffer = replay_buffer
        self._lr = lr
        self._buffer_batch_size = buffer_batch_size
        self._steps_per_epoch = steps_per_epoch
        self._eval_batch_size = eval_batch_size

        self._predictor_optimizer = torch.optim.SGD(self.predictor.parameters(), lr=self._lr)

    def train(self, runner):
        """ training loop """
        for _ in runner.step_epochs():
            for _ in range(self._steps_per_epoch):
                obj_loss, accuracy = self.train_once()
            self._log_statistics(obj_loss, accuracy)
            # save conv encoder weights!
            if self.predictor._cnn_encoder is not None:
                runner.save_state_dict(self.predictor._cnn_encoder, 'encoder')
            runner.step_itr += 1

    def train_once(self):
        """ sample data and optimize predictor """
        samples = self.replay_buffer.sample_transitions(
            self._buffer_batch_size)
        samples = dict_np_to_torch(samples)
        obj_loss = self.optimize_predictor(samples)

        eval_samples = self.replay_buffer.sample_transitions(
            self._eval_batch_size)
        eval_samples = dict_np_to_torch(eval_samples)
        accuracy = self.evaluate(eval_samples)

        return obj_loss, accuracy

    @abc.abstractmethod
    def evaluate(self, samples):
        """ evaluate accuracy of predictor """

    @abc.abstractmethod
    def optimize_predictor(self, samples):
        """ compute objective and update params """

    def _log_statistics(self, loss, accuracy):
        tabular.record('Loss', loss.item())
        tabular.record('TrainAccuracy', accuracy)

    @property
    def networks(self):
        return [self.predictor]

    def to(self, device=None):
        if device is None:
            device = global_device()
        for net in self.networks:
            net.to(device)



