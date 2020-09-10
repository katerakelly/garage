import numpy as np
import torch
import torch.nn.functional as F

from garage.torch.algos import SAC

class DiscreteSAC(SAC):
    """
    SAC with discrete actions.
    The changes from original SAC are:
    - Q-function outputs values for all actions given a state
    - policy outputs softmax distribution over all actions
    - policy update directly calculates expectation rather than using
      the reparameterization trick
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._use_automatic_entropy_tuning:
            # TODO this is hacked for the catcher env
            #self._target_entropy = -np.log(1.0 / 3.0) * 0.98
            self._target_entropy = -np.prod(
                self.env_spec.action_space.shape).item()
            print('Target entropy:', self._target_entropy)

    def _temperature_objective(self, pi, log_pi, samples_data):
        alpha_loss = 0
        if self._use_automatic_entropy_tuning:
            alpha_loss = (pi.detach() * (-(self._get_log_alpha(samples_data)) *
                          (log_pi.detach() + self._target_entropy))).mean()
        return alpha_loss

    def _actor_objective(self, samples_data, pi, log_pi):
        obs = samples_data['observation']
        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()
        min_q = torch.min(self._qf1(obs), self._qf2(obs))
        policy_objective = (pi * ((alpha * log_pi) - min_q)).mean()
        return policy_objective


    def _critic_objective(self, samples_data):
        obs = samples_data['observation']
        actions = samples_data['action']
        # convert actions from one-hot to indices
        actions = actions.max(dim=-1)[1][..., None]
        rewards = samples_data['reward'].flatten()
        terminals = samples_data['terminal'].flatten()
        next_obs = samples_data['next_observation']
        with torch.no_grad():
            alpha = self._get_log_alpha(samples_data).exp()
        # since we only have rewards for the actions we DID take
        # we can only compute the loss for those actions
        q1_pred = self._qf1(obs).gather(dim=-1, index=actions)
        q2_pred = self._qf2(obs).gather(dim=-1, index=actions)

        pi = self.policy(next_obs)[0] # batch x action_dim of softmax values
        # don't take a log of 0!
        zero_locs = pi == 0.0
        zero_locs = zero_locs.float() * 1e-8
        log_pi = torch.log(pi + zero_locs)

        min_q_values = torch.min(
            self._target_qf1(next_obs),
            self._target_qf2(next_obs)) # batch x action_dim
        # V(s) = E_pi Q(s, a) - alpha * log pi
        # take exact expectation wrt pi because we can!
        target_q_values = (pi * (min_q_values - (alpha * log_pi))).mean(dim=-1)
        with torch.no_grad():
            q_target = rewards * self._reward_scale + (
                1. - terminals) * self._discount * target_q_values
        qf1_loss = F.mse_loss(q1_pred.flatten(), q_target)
        qf2_loss = F.mse_loss(q2_pred.flatten(), q_target)

        return qf1_loss, qf2_loss

    def optimize_policy(self, samples_data):
        """
        Compute all losses
        """
        obs = samples_data['observation']
        qf1_loss, qf2_loss = self._critic_objective(samples_data)

        self._qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self._qf1_optimizer.step()

        self._qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self._qf2_optimizer.step()

        # just do this once here to avoid another forward call
        # when performing alpha update
        pi = self.policy(obs)[0]
        # don't take a log of 0!
        zero_locs = pi == 0.0
        zero_locs = zero_locs.float() * 1e-8
        log_pi = torch.log(pi + zero_locs)

        policy_loss = self._actor_objective(samples_data, pi, log_pi)
        self._policy_optimizer.zero_grad()
        policy_loss.backward()

        self._policy_optimizer.step()

        if self._use_automatic_entropy_tuning:
            alpha_loss = self._temperature_objective(pi, log_pi,
                                                     samples_data)
            self._alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self._alpha_optimizer.step()

        pi_dist = torch.distributions.Categorical(probs=pi)
        policy_entropy = pi_dist.entropy()

        return policy_loss, qf1_loss, qf2_loss, policy_entropy
