from collections import OrderedDict
import numpy as np
import math
from gym.spaces import Dict, Box
import gym
import os
import tensorflow as tf

SCRIPT_DIR = os.path.dirname(__file__)
from garage.envs.meld.reacher.sawyer_reacher import SawyerReachingEnv

##################################################################################################
##################################################################################################
##################################################################################################
##################################################################################################

class SawyerButtonsEnv(SawyerReachingEnv):

    '''
    Inserting a peg into 1 of 4 possible boxes.
    This env is a multi-task env. The reward always gets concatenated to obs.
    '''

    def __init__(self, xml_path=None, goal_site_name=None, action_mode='joint_delta_position', *args, **kwargs):

        if xml_path is None:
            xml_path = os.path.join(SCRIPT_DIR, 'assets/sawyer_buttons.xml')
        if goal_site_name is None:
            goal_site_name = 'goal_insert_site1'

        # init ids to 0 before env creation
        self.which_button = 0
        self.body_id_panel = 0
        self.site_id_goals = None

        # KEEP THESE FALSE
        self.auto_reset_task = False
        self.auto_reset_task_list = None

        super(SawyerButtonsEnv, self).__init__(xml_path=xml_path, goal_site_name=goal_site_name, action_mode=action_mode, *args, **kwargs)

        # ids of sites and bodies
        self.site_id_goals = [self.model.site_name2id('goal_insert_site1'),
                              self.model.site_name2id('goal_insert_site2'),
                              self.model.site_name2id('goal_insert_site3'),
                             ]
        self.body_id_panel = self.model.body_name2id("controlpanel")
        self.task2goal = {}
        self.train_tasks = None

    def reset_model(self):

        #### FIXED START
        angles = self.init_qpos.copy()

        # NOTICE PLAY AROUND WITH STARTING ANGLE
        # angles[0] = -0.5185 # to 0.3

        # angles[0] = np.random.uniform(-0.5185, 0.3)

        velocities = self.init_qvel.copy()
        self.set_state(angles, velocities) #this sets qpos and qvel + calls sim.forward

        return self.get_obs()


    def reset(self):

        # original mujoco reset
        self.sim.reset()
        ob = self.reset_model()

        # RESET task every episode, randomly
        if self.auto_reset_task:
            task_idx = np.random.randint(len(self.auto_reset_task_list))
            self.set_task_for_env(self.auto_reset_task_list[task_idx])

        # concatenate dummy rew=0 to the obs
        ob = np.concatenate((ob, np.array([0]), np.array([0])))

        return ob

    def get_obs_dim(self):
        # +2 for concat rews
        return len(self.get_obs()) + 2


    def step(self, action):

        self.do_step(action)

        obs = self.get_obs()
        if self.site_id_goals is None:
            reward, score, sparse_reward = 0,0,0
        else:
            reward, score, sparse_reward = self.compute_reward(get_score=True, goal_id_override=self.site_id_goals[self.which_button], button=self.which_button)
        done = False
        info = np.array([score, 0, 0, 0, 0])  # can populate with more info, as desired, for tb logging

        # append reward to obs
        obs = np.concatenate((obs, np.array([sparse_reward]), np.array([reward])))

        return obs, reward, done, info

    def goal_visibility(self, visible):

        """ Toggle the goal visibility when rendering: video should see goal, but image obs shouldn't """

        # set all sites to invisible (alpha)
        for id_num in self.site_id_goals:
            self.model.site_rgba[id_num][3]=0.0

        # set only the one visible site to visible
        if visible:
            self.model.site_rgba[self.site_id_goals[self.which_button]] = 1.0


    ##########################################
    ### These are called externally
    ##########################################

    def init_tasks(self, num_tasks, is_eval_env):

        '''
        Call this function externally, ONCE
        to define this env as either train env or test env
        and get the possible task list accordingly
        '''

        if is_eval_env:
            np.random.seed(100) #pick eval tasks as random from diff seed
        else:
            np.random.seed(101)

        x_low = 0.72
        x_high = 0.78
        y_low = 0.0
        y_high = 0.0
        z_low = z_high = -0.15

        assert num_tasks % 3 == 0

        possible_goals = []
        for task_id in range(num_tasks // 3):
            x = np.random.uniform(x_low, x_high)
            y = np.random.uniform(y_low, y_high)
            z = np.random.uniform(z_low, z_high)
            for button_idx in range(3):
                possible_goals.append([button_idx, x, y, z])

        return possible_goals


    def set_task_for_env(self, goal):

        '''
        Call this function externally,
        to reset the task
        '''

        # set which box is the correct one
        self.which_button = goal[0]

        # set the location of the panel
        self.model.body_pos[self.body_id_panel] = goal[1:]

    ##########################################
    ##########################################

    def set_auto_reset_task(self, task_list):
        self.auto_reset_task = True
        self.auto_reset_task_list = task_list

    ##########################################
    ##########################################

    def set_task_dict(self, train_tasks):
        self.train_tasks = train_tasks
        dummy_action = np.zeros(7)
        for task in train_tasks:
            self.set_task_for_env(task)
            self.do_step(dummy_action)
            goal_xyz = self.data.site_xpos[self.site_id_goals[self.which_button]].copy()
            self.task2goal[tuple(task)] = goal_xyz

    def get_relabel_tasks(self, train_tasks):
        relabel_tasks_dict = dict()
        all_buttons = [0, 1, 2]
        for t in train_tasks:
            which_button = t[0]
            xyz = t[1:]
            relabel_tasks = []
            for b in all_buttons:
                if not b == which_button:
                    relabel_tasks.append([b] + xyz)
            relabel_tasks_dict[tuple(t)] = relabel_tasks
        return relabel_tasks_dict

    def log10(self, x):
        numerator = tf.math.log(x)
        denominator = tf.math.log(tf.constant(10, dtype=numerator.dtype))
        return numerator / denominator

    def batch_relabel(self, all_data, relabel_ratio):
        # WARNING: this function needs to be CONSISTENT with out of graph reward function used DURING COLLECTION/EVAL
        # It CAN be different from the offline data's original reward

        #########################
        ### GET BASIC PARAMS
        #########################
        [step_types, observation_states, observation_pixels, observation_env_infos, actions, next_step_types, rewards, discounts] = all_data
        num_train_tasks, bs_in_episodes, episode_len, _ = observation_states.shape.as_list()
        assert num_train_tasks == len(self.train_tasks), "assume this, for ease of implementation"
        group_size = 3 # a group is a collection of tasks that has SAME panel position but DIFFERENT button. 3 buttons -> size 3
        assert num_train_tasks % group_size == 0
        num_groups = num_train_tasks // group_size
        num_episodes_per_group = group_size * bs_in_episodes

        #########################
        ### MIX THE DATA
        #########################
        # within each group, episodes will be shuffled according to this randomly sampled permutation
        all_indices = tf.constant(list(range(num_episodes_per_group)))
        group_permutations = []
        for _ in range(num_groups):
            group_permutations.append(tf.random.shuffle(all_indices))

        all_shuffled_data = []
        for data in all_data:
            grouped_data = tf.split(data, num_groups, axis=0)
            grouped_data_shape = grouped_data[0].shape.as_list()
            flattened_shape = [num_episodes_per_group] + grouped_data_shape[2:]

            shuffled_group_data = []
            for i, gd in enumerate(grouped_data):
                perm = group_permutations[i]
                flattened_gd = tf.reshape(gd, flattened_shape) # flatten
                shuffled_flattened_gd = tf.gather(flattened_gd, perm) # permute
                shuffled_gd = tf.reshape(shuffled_flattened_gd, grouped_data_shape) # back to orig dims
                shuffled_group_data.append(shuffled_gd)

            shuffled_data = tf.concat(shuffled_group_data, axis=0)
            all_shuffled_data.append(shuffled_data)

        # then the shuffled data will be re-combined with original data, following some probability
        """
                         5 eps/task
                |  task1: 1 1 1 1 1                    1 3 2 1 3            * each number represents an episode
          group1|  task2: 2 2 2 2 2    - shuffling ->  3 2 1 3 2            * here only 1 group is shown. There are more
                |  task3: 3 3 3 3 3                    2 1 3 2 1              but operation is the same
                            data                     shuffled_data
                                              (obtained in the code above)

                          0 1 0 0 0                    1 0 1 1 1
                    mask  1 0 0 1 0          inv_mask  0 1 1 0 1
                          0 0 0 0 1                    1 1 1 1 0

          combined_data = mask * data + inv_mask * shuffled_data

        Detail:
        (1-relabel_ratio) is the desired ratio of episodes that stays in its original task after re-combining
        let thresh be the probability of 1 when sampling the mask
        (1-relabel_ratio) = thresh * 1 + (1-thresh) * (1/3), since in expectation 1/3 of episodes will stay inside the same task after shuffling
        thresh = 1 - 3/2 * relabel_ratio
        this also means that relabel_ratio needs to be in [0, 2/3]

        Notice: because of this re-combining scheme, one episode can potentially appear multiple times in combined_data

        """

        assert 0 <= relabel_ratio <= 2/3
        thresh = 1 - 3/2 * relabel_ratio

        # generate mask of different dim
        base_mask_shape = [num_train_tasks, bs_in_episodes]
        rand_num = tf.random.uniform(shape=base_mask_shape, maxval=1, dtype=tf.dtypes.float16)
        base_mask = rand_num < thresh
        max_dim = max([len(d.shape.as_list()) for d in all_data])
        bm = base_mask
        base_masks = [bm]
        for i in range(max_dim - 2):
            bm = tf.expand_dims(bm, axis=-1)
            base_masks.append(bm)

        all_combined_data = []
        for data, shuffled_data in zip(all_data, all_shuffled_data):
            data_shape = data.shape.as_list()
            additional_dim = len(data_shape) - 2
            mask = tf.broadcast_to(base_masks[additional_dim], data_shape)
            mask = tf.cast(mask, data.dtype)
            inv_mask = 1 - mask

            combined_data = mask * data + inv_mask * shuffled_data
            all_combined_data.append(combined_data)

        ####################
        ### RELABEL REWARD
        ####################
        [c_step_types, c_observation_states, c_observation_pixels, c_observation_env_infos, c_actions, c_next_step_types, c_rewards, c_discounts] = all_combined_data

        # relabeling only changes last two entries in c_observation_states, c_rewards is unused thus not going to be changed
        # relabeling only depend on entries of c_observation_states that is the current ee_xyz
        all_goals = []
        # Notice: below we are assuming tensors like c_observation_states has its task dimension (dim0) follows the same
        # sequence of tasks as in experiences_as_tensor in meld_agent. Thus we should no longer shuffle/subsample task
        # dimension in the meld_agent code or anywhere else.
        for t in self.train_tasks:
            all_goals.append(self.task2goal[tuple(t)])
        all_goals = np.array(all_goals)
        goals_xyz = tf.constant(all_goals, dtype=tf.float32) # [15, 3]
        goals_xyz = tf.expand_dims(goals_xyz, axis=1)
        goals_xyz = tf.expand_dims(goals_xyz, axis=1) # [15, 1, 1, 3]
        goals_xyz = tf.broadcast_to(goals_xyz, [num_train_tasks, bs_in_episodes, episode_len] + [3]) # [15, 2, 41, 3]
        all_xyz = c_observation_states[:, :, :, -5:-2] # [15, 2, 41, 3]

        dist = 5 * tf.norm(all_xyz-goals_xyz, axis=-1) # [15, 2, 41]

        # dense reward
        offset = - (-(self.truncation_dist ** 2 + math.log10(self.truncation_dist ** 2 + 1e-5)))
        # original code: reward = -(dist ** 2 + math.log10(dist ** 2 + 1e-5)) + offset
        reward = -1 * (tf.math.square(dist) + self.log10(tf.math.square(dist) + 1e-5)) + offset

        # sparse reward
        sparse_zeros = tf.zeros_like(reward)
        mask = dist < self.truncation_dist # if True: dense region
        mask = tf.cast(mask, reward.dtype)
        # original code: sparse_reward = reward + self.sparse_margin
        sparse_reward = mask * (self.sparse_margin + reward) + sparse_zeros # adding zero is not necessary, but we might make sparse_reward=1 in the future
        rewards = tf.stack([sparse_reward, reward], axis=-1)

        pure_mixed_obs = c_observation_states[:, :, :, :-2]
        relabeled_c_observation_states = tf.concat([pure_mixed_obs, rewards], axis=-1)

        return [c_step_types, relabeled_c_observation_states, c_observation_pixels, c_observation_env_infos, c_actions, c_next_step_types, c_rewards, c_discounts]

    def relabel(self, time_step, action_step, next_time_step, relabel_tasks):
        relabeled_trajs = []

        for task in relabel_tasks:
            goal_xyz = self.task2goal[tuple(task)] # (3,) array
            ee_xyz = time_step.observation['state'][0, -5:-2] # (3,) tensor

            ### DISTANCE
            dist = 5 * tf.norm(ee_xyz - goal_xyz)

            sparse_dist = tf.math.minimum(dist, self.truncation_dist)  # if dist too large: return the reward at truncate_dist

            # dense reward
            # use GPS cost function: log + quadratic encourages precision near insertion
            offset = (-(self.truncation_dist ** 2 + math.log10(self.truncation_dist ** 2 + 1e-5)))
            ### GPS func
            reward = -(dist ** 2 + self.log10(dist ** 2 + 1e-5)) - offset

            # sparse reward
            # offset the whole reward such that when dist>truncation_dist, the reward will be exactly 0
            sparse_reward = -(sparse_dist ** 2 + self.log10(sparse_dist ** 2 + 1e-5)) - offset

            rewards = tf.stack([sparse_reward, reward], axis=0)
            rewards = tf.expand_dims(rewards, axis=0)

            # modify
            state = time_step.observation['state'][:, :-2]
            relabeled_state = tf.concat([state, rewards], axis=-1)
            time_step.observation['state'] = relabeled_state

            '''
            traj = Trajectory(step_type=time_step.step_type,
                              observation=time_step.observation, # this is modified: since the reward we use is in observation
                              action=action_step.action,
                              policy_info=action_step.info,
                              next_step_type=next_time_step.step_type,
                              reward=next_time_step.reward, # this is NOT modified since it is only used for eval performance, which is not needed for relabeled data
                              discount=next_time_step.discount)
            '''

            relabeled_trajs.append(traj)

        return relabeled_trajs

