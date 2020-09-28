import abc
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from garage.torch import global_device
from garage.torch.modules import MLPModule
from garage.np import compute_perclass_accuracy


class Predictor(abc.ABC, nn.Module):
    """
    Base class for prediction module that consists of CNN encoder
    and head modules
    """
    def __init__(self,
                 cnn_encoder,
                 head,
                 action_dim=None,
                 train_cnn=True):
        super().__init__()
        self.cnn_encoder = cnn_encoder
        self.head = head
        self._action_dim = action_dim
        self._train_cnn = train_cnn

    def get_trainable_params(self):
        """ return parameters to be trained by optimizer """
        if self._train_cnn:
            return list(self.parameters())
        else:
            return list(self.head.parameters())

    def forward(self, inputs):
        """ pass inputs through cnn encoder (if it exists) then head (if it exists) """
        if self.cnn_encoder is not None:
            # TODO this might be slower than reshaping images into batch dim
            feats = [self.cnn_encoder(in_) for in_ in inputs]
        else:
            feats = inputs
        feat = torch.cat(feats, dim=-1)
        if self.head is not None:
            return self.head(feat)
        else:
            return feat

    def compute_loss(self, samples_data):
        """ compute the prediction loss """

    def evaluate(self, samples_data):
        """ evaluate predictor with appropriate metrics """


class CPC(Predictor):
    """
    Maximizes I(z_{t+1}, z_t)
    """
    def __init__(self, cnn_encoder, **kwargs):
        super().__init__(cnn_encoder, head=None, **kwargs)
        z_dim = self.cnn_encoder.output_dim
        self.W = nn.Parameter(torch.rand(z_dim, z_dim)) # optimized
        self._loss = nn.CrossEntropyLoss()

    def get_trainable_params(self):
        return list(self.parameters())

    def forward(self, inputs):
        """
        embed obs -> z_t and next_obs -> z_{t+1} with cnn
        """
        obs, next_obs = inputs
        if self.cnn_encoder is not None:
            obs_feat = self.cnn_encoder(obs)
            next_obs_feat = self.cnn_encoder(next_obs)
        else:
            obs_feat = obs
            next_obs_feat = next_obs
        return obs_feat, next_obs_feat

    def compute_logits(self, z_a, z_pos):
        """
        copied from: https://github.com/MishaLaskin/curl
        Uses logits trick for CURL:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

    def prepare_data(self, samples_data):
        obs = samples_data['observation']
        next_obs = samples_data['next_observation']
        if torch.isnan(obs).byte().any() or torch.isnan(next_obs).byte().any():
            print('an input is NaN')
            raise Exception
        if (np.abs(obs) > 1).any() or (np.abs(next_obs) > 1).any():
            print(obs)
            print(next_obs)
            raise Exception
        return [obs, next_obs]

    def compute_loss(self, samples_data):
        """
        idea is to sample a batch of (ob, next_ob), encode them,
        and then for each ob, use the rest of the batch as the
        negative example next_ob in the contrastive loss
        """
        data = self.prepare_data(samples_data)
        anchor, positives = self.forward(data)
        logits = self.compute_logits(anchor, positives)
        labels = torch.arange(logits.shape[0]).long().to(global_device())
        loss = self._loss(logits, labels)
        avg_logit = torch.abs(logits).mean().item()
        w_norm = torch.abs(self.W).mean().item()
        anchor_norm = torch.abs(anchor).mean().item()
        positives_norm = torch.abs(positives).mean().item()
        return {'CELoss': loss}, {'AvgLogit': avg_logit, 'WNorm': w_norm, 'AnchorNorm': anchor_norm, 'PositiveNorm': positives_norm}

    def evaluate(self, samples_data):
        data = self.prepare_data(samples_data)
        anchor, positives = self.forward(data)
        logits = self.compute_logits(anchor, positives)
        preds = torch.argmax(logits, dim=-1)
        device = global_device()
        labels = torch.arange(logits.shape[0]).long().to(device)
        correct = (preds == labels).sum().item()
        return {'accuracy': correct / len(labels)}


class ForwardMI(CPC):
    """
    UL algorithm that maximizes I(z_{t+1}; z_t, a_t)
    """
    def __init__(self, cnn_encoder, **kwargs):
        super().__init__(cnn_encoder, **kwargs)
        # NOTE hard-coded for catcher env
        self.context_head = MLPModule(input_dim=self.cnn_encoder.output_dim,
                              output_dim=self.cnn_encoder.output_dim * self._action_dim,
                              hidden_sizes=[],
                              hidden_nonlinearity=None)

    def get_trainable_params(self):
        return list(self.parameters())

    def forward(self, inputs):
        """
        inputs: [obs, next_obs, action]
        encode obs and next_obs with cnn into z and z'
        concat [a, z], embed to get anchor c
        return: anchor c, positives z'
        """
        obs, next_obs, action = inputs
        if self.cnn_encoder is not None:
            obs_feat = self.context_head(self.cnn_encoder(obs)).view(-1, self.cnn_encoder.output_dim, self._action_dim)
            next_obs_feat = self.cnn_encoder(next_obs)
        else:
            obs_feat = obs
            next_obs_feat = next_obs
        # incorporate action by masking context feature with 1-hot action
        context = torch.einsum('bij, bjk->bik', obs_feat, action[..., None]) # batch x feat x 1
        context = context.squeeze()
        return context, next_obs_feat

    def prepare_data(self, samples_data):
        obs = samples_data['observation']
        next_obs = samples_data['next_observation']
        actions = samples_data['action']
        return [obs, next_obs, actions]


class InverseMI(Predictor):
    """
    UL algorithm that trains an inverse model p(a | o_t, o_t+1)
    """
    def __init__(self, cnn_encoder, head, discrete=True, information_bottleneck=False, kl_weight=1.0, **kwargs):
        super().__init__(cnn_encoder, head, **kwargs)
        self._discrete = discrete
        self._information_bottleneck = information_bottleneck
        if self._information_bottleneck:
            # use the same var for every datapoint, trainable
            self._var = nn.Parameter(torch.zeros(self.cnn_encoder.output_dim) -3.)
            self._kl_weight = kl_weight
        if self._discrete:
            self._loss = nn.CrossEntropyLoss()
        else:
            self._loss = F.mse_loss

    def get_trainable_params(self):
        return list(self.parameters())

    def compute_loss(self, samples_data):
        obs = samples_data['observation']
        next_obs = samples_data['next_observation']
        actions = samples_data['action']

        pred_actions = self.forward([obs, next_obs])
        if self.cnn_encoder is not None:
            obs_feat = self.cnn_encoder(obs)
            next_obs_feat = self.cnn_encoder(next_obs)
        else:
            obs_feat = obs
            next_obs_feat = next_obs

        # optionally compute IB loss
        # assume output of cnn is diagonal gaussian
        if self._information_bottleneck:
            latent_dim = self.cnn_encoder.output_dim
            prior = torch.distributions.Normal(
                torch.zeros(latent_dim).to(global_device()),
                torch.ones(latent_dim).to(global_device()))
            kl_div_sum = 0
            samples = []
            z_mean, z_std = [], []
            for feat in [obs_feat, next_obs_feat]:
                std = torch.sqrt(F.softplus(self._var))
                posteriors = [torch.distributions.Normal(mu, std) for mu in torch.unbind(feat)]
                samples.append(torch.stack([post.rsample() for post in posteriors]))
                kl_divs = [torch.distributions.kl.kl_divergence(post, prior) for post in posteriors]
                kl_div_sum += torch.mean(torch.stack(kl_divs))
                # get mean and std of posteriors for logging
                z_mean.append(torch.stack([torch.abs(post.mean) for post in posteriors]).mean())
                z_std.append(torch.stack([post.stddev for post in posteriors]).mean())

            feat = torch.cat(samples, dim=-1)
        else:
            feat = torch.cat([obs_feat, next_obs_feat], dim=-1)

        # predict action from combined image features
        pred_actions = self.head(feat)

        # compute the action-prediction loss
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
        if self._information_bottleneck:
            z_mean = torch.mean(torch.stack(z_mean)).item()
            z_std = torch.mean(torch.stack(z_std)).item()
            return {'KL': kl_div_sum * self._kl_weight, 'CELoss': loss}, {'ZMean': z_mean, 'ZStd': z_std}
        else:
            return {'CELoss': loss}, {}

    def evaluate(self, samples_data):
        """ return dict of (stat name, value) """
        obs = samples_data['observation']
        next_obs = samples_data['next_observation']
        actions = samples_data['action']

        if not self._discrete:
            raise NotImplementedError
        pred_actions = torch.argmax(self.forward([obs, next_obs]), dim=-1)
        actions = torch.argmax(actions, dim=-1) # convert 1-hot to scalar
        eval_dict = {}
        total_acc = (actions == pred_actions).sum().item() / len(actions)
        eval_dict['avg_accuracy'] = total_acc

        # also break out accuracy per action
        pred_actions = pred_actions.detach().cpu().numpy()
        actions = actions.cpu().numpy()
        action_vals = list(range(3))
        accs = compute_perclass_accuracy(pred_actions, actions, action_vals)
        d = dict([(f'accuracy_{val}', acc) for val, acc in zip(action_vals, accs)])
        eval_dict.update(d)

        return eval_dict


class RewardDecoder(Predictor):
    """
    classify the reward given the current observation
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 2% of dataset is neg reward, 1% is positive!
        #loss_weight = torch.Tensor([0.29, 0.01, 0.7])
        loss_weight = torch.Tensor([1.0, 1.0, 1.0])
        self._loss = nn.CrossEntropyLoss(weight=loss_weight)

    def get_trainable_params(self):
        return list(self.parameters())

    def remap_reward(self, reward):
        # map the raw reward to three classes
        reward[reward == -1.0] = 0
        reward[reward == 0.0] = 1
        reward[reward == 1.0] = 2
        return reward

    def compute_loss(self, samples_data):
        # predict reward from NEXT obs since this is what it depends on in this env
        obs = samples_data['next_observation']
        reward = samples_data['reward']
        reward = self.remap_reward(reward).long()

        # compute the loss
        pred = self.forward([obs])
        loss = self._loss(pred, reward.flatten())

        return {'CELoss': loss}, {}

    def evaluate(self, samples_data):
        obs = samples_data['observation']
        reward = samples_data['reward']
        reward = self.remap_reward(reward).cpu().numpy().flatten()
        pred = torch.argmax(self.forward([obs]), dim=-1).detach().cpu().numpy()

        # separate metric by reward value (since there are only a few)
        reward_vals = list(range(3))
        accs = compute_perclass_accuracy(pred, reward, reward_vals)
        eval_dict = dict([(f'reward_{r}', acc) for r, acc in zip(reward_vals, accs)])
        return eval_dict


class StateDecoder(Predictor):
    """
    Predict ground truth state from current observation
    """
    def compute_loss(self, samples_data):
        obs = samples_data['observation']
        target = samples_data['env_info']

        # compute the loss
        pred = self.forward([obs])
        loss = F.mse_loss(pred.flatten(), target.flatten())

        return {'MSELoss': loss}, {}

    def evaluate(self, samples_data):
        """ report mean squared error """
        obs = samples_data['observation']
        state = samples_data['env_info']

        # TODO hard-coded for catcher env
        pred_state = self.forward([obs])
        # break out into error predicting agent loc and fruit loc
        agent_mse = F.mse_loss(pred_state[..., :2].flatten(), state[..., :2].flatten())
        fruit_mse = F.mse_loss(pred_state[..., 2:4].flatten(), state[..., 2:4].flatten())
        stats = {'AgentMSE': agent_mse.item(), 'FruitMSE': fruit_mse.item()}
        if self._action_dim > 3:
            gripper_mse = F.mse_loss(pred_state[..., -1].flatten(), state[..., -1].flatten())
            stats.update({'GripperMSE': gripper_mse.item()})
        return stats


