from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from ProtoPSAC import ProtoSAC, CustomSACPolicy
from car_racing import CarRacingVFD
from evaluation import evaluate_policy
import matplotlib.pyplot as plt
# Train SAC on Car Racing environment

def make_env(render=None):
    """
    Utility function for multiprocessed env.
    
    :param env_name: (str) the environment ID
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = CarRacingVFD()#render_mode = "human")
        return env
    return _init

def train():
    num_envs = 8
    envs = [make_env() for i in range(num_envs)]
    env = SubprocVecEnv(envs)
    env = VecTransposeImage(env)

    # Initialize the SAC model with a CNN policy
    # sac_model = SAC("CnnPolicy", env, verbose=1, device="cuda")
    # ppsac_model = ProtoSAC(CustomSACPolicy, env, verbose=1, device="cuda")

    # Training
    # Create test environment
    test_env = DummyVecEnv([make_env()])
    test_env = VecTransposeImage(test_env)
    # model_cls = [SAC, ProtoSAC]
    training_iters = 10 # aiming for 10,000,000 training steps
    training_seed = 1
    eval_seed = 99
    data = {
        "model": [],
        "reward": [],
    }

    for i in range(2):
        total_training_steps = 0
        model = None
        # if i == 0:
        #     model = SAC("CnnPolicy", env, verbose=1, device="cuda")
        #     # model = SAC.load("sac_carracing_model_1400000", env=env)
        #     # total_training_steps = 1400000
        # if i == 1:
        #     model = ProtoSAC(CustomSACPolicy, env, verbose=1, device="cuda")
        model = ProtoSAC(CustomSACPolicy, env, verbose=1, device="cuda")

        env.seed(training_seed)
        test_env.seed(eval_seed)
        for _ in range(training_iters):
            time_steps = 100000  # Adjust based on computational resources
            model.learn(total_timesteps=time_steps)
            total_training_steps += time_steps

            # Save the model
            algorithm_name = "proto" if isinstance(model, ProtoSAC) else ""
            algorithm_name = algorithm_name + f"sac_carracing_model_{total_training_steps}"
            model.save(algorithm_name)

            # Evaluation
            # model = ProtoSAC.load("sac_carracing_model_15000", env=env)
            # model = SAC.load("sac_carracing_model_100000", env=env)
            eval_seeds = [93, 122, 854]
            time_steps_list = []
            rewards_list = []
            # for eval_seed in eval_seeds:
            #     test_env.seed(eval_seed)
            #     rewards_env = []
            #     for _ in range(0, time_steps, 100000):
            #         rewards = evaluate_policy(model, test_env, n_eval_episodes=5)[0] #, render=True)
            #         rewards_env.append(rewards)
            #         print(f"EVAL REWARD WITH SEED {eval_seed}")
            #         print(rewards_env)

            #     time_steps_list.append(list(range(0, time_steps, 100000)))
            #     rewards_list.append(rewards_env)

            # for i in range(len(eval_seeds)):
            #     plt.plot(time_steps[i], rewards_list[i], label = f"Eval seed {eval_seeds[i]}")

            # plt.xlabel("Number of environment interactions")
            # plt.ylabel("success")
            # plt.title("Number of Environment Interactions vs Success")
            # plt.legend()
            # plt.show()

            # print(reward)

            # store data
            # data["model"].append(algorithm_name)
            # data["reward"].append(reward)
            # dataframe = pd.DataFrame(data)
            # dataframe.to_pickle("training_data4.pkl")

    # Close the environment
    env.close()

if __name__ == '__main__':
    train()

# ramprasad plots
# time_steps= 5000
# time_steps_list = []
# rewards_list = []
# eval_seeds = [93, 100, 134]
# env = make_env()
# for eval_seed in eval_seeds:
#     env.seed(eval_seed)
#     rewards_env = []
#     for time_step in range(0,time_steps, 100000):
#         rewards = evaluate_policy(model, env, n_eval_episodes=5) #, render=True)
#         rewards_env.append(rewards)
#         print(f"EVAL REWARD WITH SEED {eval_seed}")
#         print(rewards_env)
#     time_steps_list.append(list(range(1, time_steps+1)))
#     rewards_list.append(rewards_env)

# for i in range(len(eval_seeds)):
#     plt.plot(time_steps[i], rewards_list[i], label = f"Eval seed {eval_seeds[i]}")

# plt.xlabel("Number of environment interactions")
# plt.ylabel("success")
# plt.title("Number of Environment Interactions vs Success")
# plt.legend()
# plt.show()