import tensorflow as tf
import functools

from garage.tf.models import CNNModel
from garage.tf.models import Model
from garage.tf.models import GaussianGRUModel

class MELDCnnModel(CNNModel):
  """Feature extractor: images-->features """

  def __init__(self, base_depth, name=None, double_camera=False):
    if not double_camera:
        layers = []
        layers.append((base_depth, (5, 5)))
        layers.append((4 * base_depth, (3, 3)))
        layers.append((4 * base_depth, (3, 3)))
        layers.append((8 * base_depth, (3, 3)))
        layers.append((8 * base_depth, (4, 4)))
        strides = (2, 2, 2, 2, 1)
        padding = ("SAME", "SAME", "SAME", "SAME", "VALID")
    else:
        raise NotImplementedError
    super().__init__(filters=layers,
                     strides=strides,
                     padding=padding,
                     name=name)
    '''
    if not self.double_camera:
      self.conv1 = conv(base_depth, 5, 2)
      self.conv2 = conv(2 * base_depth, 3, 2)
      self.conv3 = conv(4 * base_depth, 3, 2)
      self.conv4 = conv(8 * base_depth, 3, 2)
      self.conv5 = conv(8 * base_depth, 4, padding="VALID")
    else:
      self.conv1 = conv(base_depth, (10, 5), 2)  # conv: filters, kernel_size, stride
      self.conv2 = conv(2 * base_depth, (6, 3), 2)
      self.conv3 = conv(4 * base_depth, (6, 3), 2)
      self.conv4 = conv(8 * base_depth, (6, 3), 2)
      self.conv5 = conv(8 * base_depth, (8, 4), padding="VALID")
    '''

class MELDModel(Model):
    """ Feature extractor + GRU Policy """
    def __init__(self, obs_dim, act_dim, hidden_dim=32, double_camera=False, name='MELDModel'):
        super().__init__(name)
        self._obs_dim = obs_dim
        self._action_dim = act_dim
        self._rest_input_dim = self._action_dim + 1 + 1
        self._image_obs_dim = self._obs_dim - self._rest_input_dim # image space
        # TODO hard-coded sizes here
        self._conv_feat_dim = 256
        if not double_camera:
            self._image_3d_obs_dim = (-1, 64, 64, 3)
        else:
            raise NotImplementedError

        self.cnn_model = MELDCnnModel(base_depth=32, double_camera=double_camera)
        self.gru_model = GaussianGRUModel(
                output_dim=self._action_dim,
                hidden_dim=hidden_dim,
                name='GaussianGRUModel')

        # input and output specs are the same as regular gassuain gru model
    def network_input_spec(self):
        """Network input spec.

        Returns:
            list[str]: Name of the model inputs, in order.

        """
        return ['full_input', 'step_input', 'step_hidden_input']

    def network_output_spec(self):
        """Network output spec.

        Returns:
            list[str]: Name of the model outputs, in order.

        """
        return [
            'dist', 'step_mean', 'step_log_std', 'step_hidden', 'init_hidden'
        ]

    def _build(self, state_input, step_input_var, step_hidden_var, name=None):
        '''
        state_input: time-series input
        step_input: single step input
        step_hidden: step hidden state
        '''
        name = name or ''
        # split the input var into obs and rest
        # batch and time dims should remain untouched
        batch, time = tf.shape(state_input)[0], tf.shape(state_input)[1]

        ### first take care of time series input
        # time series input is batch x time x feat
        obs_series_input = tf.gather(state_input, list(range(self._image_obs_dim)), axis=-1)
        # conv input must be rank 4 so squash batch and time dims
        obs_series_input = tf.reshape(obs_series_input, self._image_3d_obs_dim)
        rest_series_input = tf.gather(state_input, list(range(self._image_obs_dim, self._obs_dim)), axis=-1)
        # pass the obs input through the cnn
        # output is batch x feat
        # TODO why are there a list of things here??
        cnn_series_output = self.cnn_model.build(obs_series_input, name='cnn_' + name)[-1]
        # explode batch and time dimensions back
        # TODO hard-coded shape here
        cnn_series_output = tf.reshape(cnn_series_output, (batch, time, self._conv_feat_dim))
        # concat cnn feature with rest input
        gru_series_input = tf.concat([cnn_series_output, rest_series_input], axis=-1)

        ### now take care of the step input
        # step input is batch x feat
        batch_step = tf.shape(step_input_var)[0]
        obs_step_input = tf.gather(step_input_var, list(range(self._image_obs_dim)), axis=-1)
        obs_step_input = tf.reshape(obs_step_input, self._image_3d_obs_dim)
        rest_step_input = tf.gather(step_input_var, list(range(self._image_obs_dim, self._obs_dim)), axis=-1)
        cnn_step_output = self.cnn_model.build(obs_step_input, name='cnn_step_' + name)[-1]
        cnn_step_output = tf.reshape(cnn_step_output, (batch_step, self._conv_feat_dim))
        gru_step_input = tf.concat([cnn_step_output, rest_step_input], axis=-1)


        #### pass both into the gru
        (dist, step_mean, step_log_std, step_hidden,
            init_hidden) = self.gru_model.build(gru_series_input, gru_step_input,
                                                step_hidden_var, name='gru_step_' + name).outputs
        return dist, step_mean, step_log_std, step_hidden, init_hidden


