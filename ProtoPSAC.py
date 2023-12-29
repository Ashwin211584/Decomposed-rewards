import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from ProtoPNet.model import PPNet, construct_PPNet
from stable_baselines3 import SAC
from stable_baselines3.sac.policies import CnnPolicy
from stable_baselines3.common.policies import ContinuousCritic
import torch
from stable_baselines3.sac.policies import SACPolicy, ContinuousCritic
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.evaluation import evaluate_policy

import torch as th
from torch.nn import functional as F
from stable_baselines3.common.utils import get_parameters_by_name, polyak_update
import numpy as np

from typing import List, Dict, Any
from stable_baselines3.common.buffers import ReplayBuffer

import io
import pathlib
import sys
import time
import warnings
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.buffers import DictReplayBuffer, ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl, save_to_pkl
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.type_aliases import (
    DictReplayBufferSamples,
    DictRolloutBufferSamples,
    ReplayBufferSamples,
    RolloutBufferSamples,
)
from typing import Any, Callable, Dict, List, NamedTuple

class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor
    decomposed_rewards: th.Tensor

class DVFReplayBuffer(ReplayBuffer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        num_reward_components = 4
        self.infos = np.zeros((self.buffer_size, self.n_envs, num_reward_components), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs).copy()

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs).copy()
        else:
            self.next_observations[self.pos] = np.array(next_obs).copy()

        self.actions[self.pos] = np.array(action).copy()
        self.rewards[self.pos] = np.array(reward).copy()
        self.dones[self.pos] = np.array(done).copy()

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])


        # add DVF from info to replay buffer
        keys = ["time", "movement", "new_tile", "out_of_bounds"]
        reward_components_dict = infos[0].copy()
        reward_components = np.array([reward_components_dict[k] for k in keys], dtype=np.float32)
        self.infos[self.pos] = reward_components


        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def _get_samples(self, batch_inds: np.ndarray, env: Optional[VecNormalize] = None) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :], env)
        else:
            next_obs = self._normalize_obs(self.next_observations[batch_inds, env_indices, :], env)

        data = (
            self._normalize_obs(self.observations[batch_inds, env_indices, :], env),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self._normalize_reward(self.rewards[batch_inds, env_indices].reshape(-1, 1), env),
            self.infos[batch_inds, env_indices, :]
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

# Implement SAC with ProtoPNet critics

class CustomCritic(ContinuousCritic):
    def __init__(self, observation_space, action_space, net_arch, features_extractor_class, features_extractor_kwargs):
        super(CustomCritic, self).__init__(
            observation_space, 
            action_space, 
            net_arch, 
            features_extractor_class(observation_space, **features_extractor_kwargs), 
            3, # This sets input dimensions of an MLP critic used in parent class - not important
            share_features_extractor = False
        )
        # TODO: update num_classes to number of reward components
        self.num_reward_components = 4

        # Initialize PPNet with correct dimensions, replace this with the correct initialization as per your requirements
        self.critic = construct_PPNet(base_architecture='resnet18', pretrained=False, img_size=96, prototype_shape=(self.num_reward_components*10, 512, 1, 1), num_classes=self.num_reward_components)
        self.critic2 = construct_PPNet(base_architecture='resnet18', pretrained=False, img_size=96, prototype_shape=(self.num_reward_components*10, 512, 1, 1), num_classes=self.num_reward_components)
        self.critic.cuda()
        self.critic.cuda()
        self.construct_optimizers()
        self.count = -1

    def construct_optimizers(self):
        default_learning_rate = 3e-3
        # define optimizers for critic 1
        # from settings import joint_optimizer_lrs, joint_lr_step_size
        joint_optimizer_specs = \
        [{'params': self.critic.features.parameters(), 'lr': default_learning_rate, 'weight_decay': 1e-3}, # bias are now also being regularized
        {'params': self.critic.add_on_layers.parameters(), 'lr': default_learning_rate, 'weight_decay': 1e-3},
        {'params': self.critic.prototype_vectors, 'lr': default_learning_rate},
        ]
        self.critic_joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
        # joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=5, gamma=0.1)

        # from settings import warm_optimizer_lrs
        # warm_optimizer_specs = \
        # [{'params': self.critic.add_on_layers.parameters(), 'lr': default_learning_rate, 'weight_decay': 1e-3},
        # {'params': self.critic.prototype_vectors, 'lr': default_learning_rate},
        # ]
        warm_optimizer_specs = \
        [
            {'params': self.critic2.features.parameters(), 
             'lr': default_learning_rate, 'weight_decay': 1e-3},
        ]
        self.critic_warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

        # from settings import last_layer_optimizer_lr
        last_layer_optimizer_specs = [{'params': self.critic.last_layer.parameters(), 'lr': default_learning_rate}]
        self.critic_last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

        # define optimizers for critic 2
        joint_optimizer_specs = \
        [{'params': self.critic2.features.parameters(), 'lr': default_learning_rate, 'weight_decay': 1e-3}, # bias are now also being regularized
        {'params': self.critic2.add_on_layers.parameters(), 'lr': default_learning_rate, 'weight_decay': 1e-3},
        {'params': self.critic2.prototype_vectors, 'lr': default_learning_rate},
        ]
        self.critic2_joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
        # joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=5, gamma=0.1)

        # from settings import warm_optimizer_lrs
        warm_optimizer_specs = \
        [
            # {'params': self.critic2.add_on_layers.parameters(), 
            {'params': self.critic2.features.parameters(), 
             'lr': default_learning_rate, 'weight_decay': 1e-3},
        ]
        self.critic2_warm_optimizer = torch.optim.Adam(warm_optimizer_specs)

        # from settings import last_layer_optimizer_lr
        last_layer_optimizer_specs = [{'params': self.critic2.last_layer.parameters(), 'lr': default_learning_rate}]
        self.critic2_last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)

    def forward(self, obs, actions):
        # Original obs shape: [256, 3, 96, 96]
        obs_shape = obs.shape

        # Reshape actions to [256, 3, 1, 1]
        actions = actions.view(-1, 3, 1, 1)

        # Expand actions to [256, 3, 96, 96]
        actions_expanded = actions.expand(-1, -1, obs_shape[2], obs_shape[3])

        # Concatenate obs and actions
        input = torch.cat([obs, actions_expanded], dim=1)  # Resulting shape: [256, 6, 96, 96]

        # Extract features using PPNet
        critic_logits, critic_min_distances = self.critic(input)
        critic_logits2, critic_min_distances2 = self.critic2(input)

        # Store distances for backwards step
        self.critic_min_distances = critic_min_distances
        self.critic_min_distances2 = critic_min_distances2

        # Return q value logits
        return critic_logits, critic_logits2

    def backward(self, q_values, target_q_values):
        # calculate Q value MSE
        mse_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in q_values)
        device = mse_loss.device
        # calculate separation and cluster costs
        max_dist = (self.critic.prototype_shape[1]
                    * self.critic.prototype_shape[2]
                    * self.critic.prototype_shape[3])

        # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
        total_cluster_cost = torch.zeros([1]).to(device)
        total_separation_cost = torch.zeros([1]).to(device)
        total_l1 = torch.zeros([1]).to(device)
        total_cluster_cost_2 = torch.zeros([1]).to(device)
        total_separation_cost_2 = torch.zeros([1]).to(device)
        total_l1_2 = torch.zeros([1]).to(device)
        for reward_component_index in range(self.num_reward_components):
            # critic 1
            # calculate cluster cost
            prototypes_of_correct_class = torch.t(self.critic.prototype_class_identity[:,reward_component_index]).cuda()
            inverted_distances, _ = torch.max((max_dist - self.critic_min_distances) * prototypes_of_correct_class, dim=1)
            cluster_cost = torch.mean(max_dist - inverted_distances)

            # calculate separation cost
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            inverted_distances_to_nontarget_prototypes, _ = \
                torch.max((max_dist - self.critic_min_distances) * prototypes_of_wrong_class, dim=1)
            separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

            # calculate avg cluster cost
            # TODO: why does ppnet code use dim=1 for prototypes_of_wrong_class tensor?
            avg_separation_cost = \
                torch.sum(self.critic_min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=0)#, dim=1)
            avg_separation_cost = torch.mean(avg_separation_cost)
            
            # TODO: look into l1 mask
            # if use_l1_mask:
            #     l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
            #     l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
            # else:
            #     l1 = model.module.last_layer.weight.norm(p=1)
            l1 = self.critic.last_layer.weight.norm(p=1)

            # sum cost terms accross reward terms
            total_cluster_cost = total_cluster_cost + cluster_cost
            total_separation_cost = total_separation_cost + separation_cost
            total_l1 = total_l1 + l1

            # critic 2
            # calculate cluster cost
            prototypes_of_correct_class = torch.t(self.critic2.prototype_class_identity[:,reward_component_index]).cuda()
            inverted_distances, _ = torch.max((max_dist - self.critic_min_distances) * prototypes_of_correct_class, dim=1)
            cluster_cost = torch.mean(max_dist - inverted_distances)

            # calculate separation cost
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            inverted_distances_to_nontarget_prototypes, _ = \
                torch.max((max_dist - self.critic_min_distances) * prototypes_of_wrong_class, dim=1)
            separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)

            # calculate avg cluster cost
            avg_separation_cost = \
                torch.sum(self.critic_min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=0)
            avg_separation_cost = torch.mean(avg_separation_cost)
            
            # TODO: look into l1 mask
            # if use_l1_mask:
            #     l1_mask = 1 - torch.t(model.module.prototype_class_identity).cuda()
            #     l1 = (model.module.last_layer.weight * l1_mask).norm(p=1)
            # else:
            #     l1 = model.module.last_layer.weight.norm(p=1)
            l1 = self.critic2.last_layer.weight.norm(p=1)

            # sum cost terms accross reward terms
            total_cluster_cost_2 = total_cluster_cost_2 + cluster_cost
            total_separation_cost_2 = total_separation_cost_2 + separation_cost
            total_l1_2 = total_l1_2 + l1
        
        # compute gradient and perform backwards step (alternate which layers are training)*
        # TODO: alternate layers trained*
        loss = mse_loss + 0.8 * total_cluster_cost - 0.08 * total_separation_cost + 1e-4 * total_l1
        loss2 = mse_loss + 0.8 * total_cluster_cost_2 - 0.08 * total_separation_cost_2 + 1e-4 * total_l1_2

        # alternate layers trained based on number of times backwards called (based on increments of 5 gradient steps)
        self.count += 1
        training_stage = self.count % 9000 # using just joint and last layer optimizers*

        # TODO: optimal scheduling of training stages for PPNet layers is an open research question
        if training_stage < 3000:
            # 'warm up' feature layers - original PPNet code used pretrained models
            self.critic_warm_optimizer.zero_grad()
            self.critic2_warm_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            loss2.backward()
            self.critic_warm_optimizer.step()
            self.critic2_warm_optimizer.step()
        elif training_stage < 6000 and training_stage >= 3000:
            # joint layers - train all layers but the last linear layer
            self.critic_joint_optimizer.zero_grad()
            self.critic2_joint_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            loss2.backward()
            self.critic_joint_optimizer.step()
            self.critic2_joint_optimizer.step()
        else:
            # last layer training
            self.critic_last_layer_optimizer.zero_grad()
            self.critic2_last_layer_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            loss2.backward()
            self.critic_last_layer_optimizer.step()
            self.critic2_last_layer_optimizer.step()


class CustomSACPolicy(SACPolicy):
    def _build(self, lr_schedule):
        # self._build_mlp_extractor()
        self.actor = self.make_actor()
        self.actor.optimizer = self.optimizer_class(
            self.actor.parameters(),
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )
        self.critic = self.make_critic()
        critic_parameters = list(self.critic.parameters())
        self.critic.optimizer = self.optimizer_class(
            critic_parameters,
            lr=lr_schedule(1),  # type: ignore[call-arg]
            **self.optimizer_kwargs,
        )
        self.critic_target = self.make_critic()
        # Target networks should always be in eval mode
        self.critic_target.set_training_mode(False)

    def make_critic(self):
        return CustomCritic(
            self.observation_space,
            self.action_space,
            net_arch=self.net_arch,
            features_extractor_class=self.features_extractor_class,
            features_extractor_kwargs=self.features_extractor_kwargs,
            # normalize_images=self.normalize_images
        )

    def forward(self, obs, deterministic=False):
        # Implement the forward pass for the policy (actor) using SACPolicy's methods
        return self._predict(obs, deterministic=deterministic)


class ProtoSAC(SAC):
    """
    Custom implementation of SAC with Prototypical Parts Network as critic models
    """
    # policy: CustomSACPolicy # TODO: this alone didn't work*
    policy: CustomSACPolicy
    critic: CustomCritic
    critic_target: CustomCritic

    def __init__(
        self,
        policy: Union[str, Type[SACPolicy]],
        env: Union[GymEnv, str],
        **kwargs
    ):
        if "replay_buffer_class" in kwargs:
            del kwargs["replay_buffer_class"]
        super().__init__(
            policy,
            env,
            **kwargs,
            replay_buffer_class=DVFReplayBuffer,
        )

    def train(self, gradient_steps: int, batch_size: int = 64) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update optimizers learning rate
        optimizers = [self.actor.optimizer, self.critic.optimizer]
        if self.ent_coef_optimizer is not None:
            optimizers += [self.ent_coef_optimizer]

        # Update learning rate according to lr schedule
        self._update_learning_rate(optimizers)

        ent_coef_losses, ent_coefs = [], []
        actor_losses, critic_losses = [], []

        for gradient_step in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)  # type: ignore[union-attr]

            # We need to sample because `log_std` may have changed between two gradient steps
            if self.use_sde:
                self.actor.reset_noise()

            # Action by the current actor for the sampled state
            actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
            log_prob = log_prob.reshape(-1, 1)

            ent_coef_loss = None
            if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                # Important: detach the variable from the graph
                # so we don't change it with other losses
                # see https://github.com/rail-berkeley/softlearning/issues/60
                ent_coef = th.exp(self.log_ent_coef.detach())
                ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                ent_coef_losses.append(ent_coef_loss.item())
            else:
                ent_coef = self.ent_coef_tensor

            ent_coefs.append(ent_coef.item())

            # Optimize entropy coefficient, also called
            # entropy temperature or alpha in the paper
            if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                self.ent_coef_optimizer.zero_grad()
                ent_coef_loss.backward()
                self.ent_coef_optimizer.step()

            with th.no_grad():
                # Select action according to policy
                next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                # Compute the next Q values: min over all critics targets

                # TODO: need wrapper for replay_data?
                # TODO: normalize obs?

                # take the minimum of the q value component sums for each MDP state in trainin batch
                next_q_values1, next_q_values2 = self.critic_target(replay_data.next_observations.float(), next_actions)

                # Compute row-wise sums
                sums_a = next_q_values1.sum(dim=1, keepdim=True)  # Shape: [256, 1]
                sums_b = next_q_values2.sum(dim=1, keepdim=True)  # Shape: [256, 1]

                # Compare sums and create a mask for selection
                mask = sums_a < sums_b  # Shape: [256, 1]

                # Expand mask to match the shape of the tensors
                mask_expanded = mask.expand_as(next_q_values1)  # Shape: [256, 3]

                # Use the mask to select elements
                next_q_values = torch.where(mask_expanded, next_q_values1, next_q_values2)  # Shape: [256, 3]

                # next_q_values = th.cat(self.critic_target(replay_data.next_observations.float(), next_actions), dim=1)
                # next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)   # TODO: ensure minimums of compound reward are taken

                # add entropy term
                next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                # td error + entropy term

                # use decomposed rewards to calculate target q values
                target_q_values = replay_data.decomposed_rewards + (1 - replay_data.dones) * self.gamma * next_q_values

            # Get current Q-values estimates for each critic network
            # using action from the replay buffer
            current_q_values = self.critic(replay_data.observations.float(), replay_data.actions)

            # # Compute critic loss
            # critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
            # assert isinstance(critic_loss, th.Tensor)  # for type checker
            # critic_losses.append(critic_loss.item())  # type: ignore[union-attr]

            # # Optimize the critic
            # self.critic.optimizer.zero_grad()
            # critic_loss.backward()
            # self.critic.optimizer.step()

            # train critic
            self.critic.backward(current_q_values, target_q_values)

            # Compute actor loss
            # Alternative: actor_loss = th.mean(log_prob - qf1_pi)
            # Min over all critic networks
            q_values_pi = th.cat(self.critic(replay_data.observations.float(), actions_pi), dim=1)
            min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)
            actor_loss = (ent_coef * log_prob - min_qf_pi).mean()
            actor_losses.append(actor_loss.item())

            # Optimize the actor
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target networks
            if gradient_step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                # Copy running stats, see GH issue #996
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/ent_coef", np.mean(ent_coefs))
        self.logger.record("train/actor_loss", np.mean(actor_losses))
        self.logger.record("train/critic_loss", np.mean(critic_losses))
        if len(ent_coef_losses) > 0:
            self.logger.record("train/ent_coef_loss", np.mean(ent_coef_losses))
