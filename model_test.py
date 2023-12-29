import gymnasium as gym
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

from ProtoPSAC import ProtoSAC

# # Create the environment
# env_id = "CarRacing-v2"

# def make_env():
#     return gym.make(env_id) #, render_mode="rgb_array")

# # Vectorize environment
# env = DummyVecEnv([make_env])

# # Apply VecTransposeImage to handle image observations
# env = VecTransposeImage(env)

# Evaluation
# proto_model_5000 = ProtoSAC.load("proto_sac_carracing_model", env=env)
# proto_model_10000 = ProtoSAC.load("proto_sac_carracing_model_10000", env=env)
# proto_model_15000 = ProtoSAC.load("proto_sac_carracing_model_15000", env=env)
# model_5000 = SAC.load("sac_carracing_model_5000", env=env)
# model_10000 = SAC.load("sac_carracing_model_10000", env=env)
# model_15000 = SAC.load("sac_carracing_model_15000", env=env)


def weights_equal(model_dict_a, model_dict_b):
    if model_dict_a.keys() != model_dict_b.keys():
        return False
    for key in model_dict_a:
        if not torch.equal(model_dict_a[key], model_dict_b[key]):
            return False
    return True

# # policy
# print(weights_equal(model_5000.policy.state_dict(), model_10000.policy.state_dict()))  # Should be False
# print(weights_equal(model_5000.policy.state_dict(), model_15000.policy.state_dict()))  # Should be False
# print(weights_equal(model_5000.policy.state_dict(), proto_model_5000.policy.state_dict()))  # Should be False
# print(weights_equal(model_5000.policy.state_dict(), proto_model_10000.policy.state_dict()))  # Should be False
# print(weights_equal(model_5000.policy.state_dict(), proto_model_15000.policy.state_dict()))  # Should be False
# print(weights_equal(model_10000.policy.state_dict(), model_15000.policy.state_dict()))  # Should be False
# print(weights_equal(model_10000.policy.state_dict(), proto_model_5000.policy.state_dict()))  # Should be False
# print(weights_equal(model_10000.policy.state_dict(), proto_model_10000.policy.state_dict()))  # Should be False
# print(weights_equal(model_10000.policy.state_dict(), proto_model_15000.policy.state_dict()))  # Should be False
# print(weights_equal(model_15000.policy.state_dict(), proto_model_5000.policy.state_dict()))  # Should be False
# print(weights_equal(model_15000.policy.state_dict(), proto_model_10000.policy.state_dict()))  # Should be False
# print(weights_equal(model_15000.policy.state_dict(), proto_model_15000.policy.state_dict()))  # Should be False
# print(weights_equal(proto_model_5000.policy.state_dict(), proto_model_10000.policy.state_dict()))  # Should be False
# print(weights_equal(proto_model_5000.policy.state_dict(), proto_model_15000.policy.state_dict()))  # Should be False
# print(weights_equal(proto_model_10000.policy.state_dict(), proto_model_15000.policy.state_dict()))  # Should be False

# # actor
# print(weights_equal(model_5000.actor.state_dict(), model_10000.actor.state_dict()))  # Should be False
# print(weights_equal(model_5000.actor.state_dict(), model_15000.actor.state_dict()))  # Should be False
# print(weights_equal(model_5000.actor.state_dict(), proto_model_5000.actor.state_dict()))  # Should be False
# print(weights_equal(model_5000.actor.state_dict(), proto_model_10000.actor.state_dict()))  # Should be False
# print(weights_equal(model_5000.actor.state_dict(), proto_model_15000.actor.state_dict()))  # Should be False
# print(weights_equal(model_10000.actor.state_dict(), model_15000.actor.state_dict()))  # Should be False
# print(weights_equal(model_10000.actor.state_dict(), proto_model_5000.actor.state_dict()))  # Should be False
# print(weights_equal(model_10000.actor.state_dict(), proto_model_10000.actor.state_dict()))  # Should be False
# print(weights_equal(model_10000.actor.state_dict(), proto_model_15000.actor.state_dict()))  # Should be False
# print(weights_equal(model_15000.actor.state_dict(), proto_model_5000.actor.state_dict()))  # Should be False
# print(weights_equal(model_15000.actor.state_dict(), proto_model_10000.actor.state_dict()))  # Should be False
# print(weights_equal(model_15000.actor.state_dict(), proto_model_15000.actor.state_dict()))  # Should be False
# print(weights_equal(proto_model_5000.actor.state_dict(), proto_model_10000.actor.state_dict()))  # Should be False
# print(weights_equal(proto_model_5000.actor.state_dict(), proto_model_15000.actor.state_dict()))  # Should be False
# print(weights_equal(proto_model_10000.actor.state_dict(), proto_model_15000.actor.state_dict()))  # Should be False

# # actor
# print(weights_equal(model_5000.critic.state_dict(), model_10000.critic.state_dict()))  # Should be False
# print(weights_equal(model_5000.critic.state_dict(), model_15000.critic.state_dict()))  # Should be False
# print(weights_equal(model_5000.critic.state_dict(), proto_model_5000.critic.state_dict()))  # Should be False
# print(weights_equal(model_5000.critic.state_dict(), proto_model_10000.critic.state_dict()))  # Should be False
# print(weights_equal(model_5000.critic.state_dict(), proto_model_15000.critic.state_dict()))  # Should be False
# print(weights_equal(model_10000.critic.state_dict(), model_15000.critic.state_dict()))  # Should be False
# print(weights_equal(model_10000.critic.state_dict(), proto_model_5000.critic.state_dict()))  # Should be False
# print(weights_equal(model_10000.critic.state_dict(), proto_model_10000.critic.state_dict()))  # Should be False
# print(weights_equal(model_10000.critic.state_dict(), proto_model_15000.critic.state_dict()))  # Should be False
# print(weights_equal(model_15000.critic.state_dict(), proto_model_5000.critic.state_dict()))  # Should be False
# print(weights_equal(model_15000.critic.state_dict(), proto_model_10000.critic.state_dict()))  # Should be False
# print(weights_equal(model_15000.critic.state_dict(), proto_model_15000.critic.state_dict()))  # Should be False
# print(weights_equal(proto_model_5000.critic.state_dict(), proto_model_10000.critic.state_dict()))  # Should be False
# print(weights_equal(proto_model_5000.critic.state_dict(), proto_model_15000.critic.state_dict()))  # Should be False
# print(weights_equal(proto_model_10000.critic.state_dict(), proto_model_15000.critic.state_dict()))  # Should be False

# # positives
# print(weights_equal(model_5000.policy.state_dict(), model_5000.policy.state_dict()))  # Should be False
# print(weights_equal(model_10000.policy.state_dict(), model_10000.policy.state_dict()))  # Should be False
# print(weights_equal(model_15000.policy.state_dict(), model_15000.policy.state_dict()))  # Should be False
# print(weights_equal(proto_model_5000.policy.state_dict(), proto_model_5000.policy.state_dict()))  # Should be False
# print(weights_equal(proto_model_10000.policy.state_dict(), proto_model_10000.policy.state_dict()))  # Should be False
# print(weights_equal(proto_model_15000.policy.state_dict(), proto_model_15000.policy.state_dict()))  # Should be False
# print(weights_equal(model_5000.actor.state_dict(), model_5000.actor.state_dict()))  # Should be False
# print(weights_equal(model_10000.actor.state_dict(), model_10000.actor.state_dict()))  # Should be False
# print(weights_equal(model_15000.actor.state_dict(), model_15000.actor.state_dict()))  # Should be False
# print(weights_equal(proto_model_5000.actor.state_dict(), proto_model_5000.actor.state_dict()))  # Should be False
# print(weights_equal(proto_model_10000.actor.state_dict(), proto_model_10000.actor.state_dict()))  # Should be False
# print(weights_equal(proto_model_15000.actor.state_dict(), proto_model_15000.actor.state_dict()))  # Should be False
# print(weights_equal(model_5000.critic.state_dict(), model_5000.critic.state_dict()))  # Should be False
# print(weights_equal(model_10000.critic.state_dict(), model_10000.critic.state_dict()))  # Should be False
# print(weights_equal(model_15000.critic.state_dict(), model_15000.critic.state_dict()))  # Should be False
# print(weights_equal(proto_model_5000.critic.state_dict(), proto_model_5000.critic.state_dict()))  # Should be False
# print(weights_equal(proto_model_10000.critic.state_dict(), proto_model_10000.critic.state_dict()))  # Should be False
# print(weights_equal(proto_model_15000.critic.state_dict(), proto_model_15000.critic.state_dict()))  # Should be False

# wrap eval?
from stable_baselines3.common.vec_env.vec_monitor import VecMonitor
from gymnasium.envs.box2d.car_racing import CarRacing, FrictionDetector, Car
from typing import Union, Optional
from gymnasium.error import DependencyNotInstalled, InvalidAction

from car_racing import CarRacingVFD

# Create the environment
env_id = "CarRacing-v2"

def make_env():
    # return gym.make(env_id, render_mode="human")
    return CarRacingVFD(render_mode="human")

# Vectorize environment
env = DummyVecEnv([make_env])

# Apply VecTransposeImage to handle image observations
env = VecTransposeImage(env)

# env = Monitor(env)
# env = VecMonitor(env)


sac_model_v3_500k = SAC.load("sac_carracing_v3_model_100000", env=env)

# # Run an evaluation episode
# obs = env.reset()
# done = False
# reward = 0
# while not done:
#     action, _states = sac_model_900k.predict(obs, deterministic=True)
#     obs, rewards, done, info = env.step(action)
#     reward += rewards
#     env.render()

eval_seeds = [93]#, 122, 854]
for eval_seed in eval_seeds:
    env.seed(eval_seed)
    reward = evaluate_policy(sac_model_v3_500k, env, n_eval_episodes=1, render=True)
    # env.seed(eval_seed)
    # reward2 = evaluate_policy(model_10000, env, n_eval_episodes=1) #, render=True)
    # env.seed(eval_seed)
    # reward3 = evaluate_policy(model_15000, env, n_eval_episodes=1) #, render=True)
    print(f"EVAL REWARD WITH SEED {eval_seed}")
    print(reward)
    # print(reward2)
    # print(reward3)
