# General imports
import numpy as np
import time
import csv
import os
import constants
from queue import Queue
from threading import Thread

# OpenAI Gymnasium imports
import gymnasium as gym
from gymnasium.spaces import MultiDiscrete, Discrete, Box

# StableBaselines3 imports
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# SC2 API imports
from sc2.data import Difficulty, Race
from sc2.main import run_game
from sc2.player import Bot, Computer
from sc2 import maps

# Bot
from VoidRayBot import VRBot

# Global variables to pick the right experiment and WandB project.
mapName = "Simple64"
episode_reward_list = []

# Change the comments in the following two lines to create a new model 
model_name = f"{int(time.time())}"
#model_name = 1712752179

models_dir = f"models/{model_name}/"

# This is the thread that holds the queues and runs the game
class GameThread(Thread):
    def __init__(self) -> None:
        super().__init__()
        self.action_in = Queue()
        self.result_out = Queue()
    
    def run(self) -> None:
        self.bot = VRBot(action_in=self.action_in, result_out=self.result_out)
        print("starting game.")
        result = run_game(  # run_game is a function that runs the game.
            maps.get(mapName), # the map we are playing on
            [Bot(Race.Protoss, self.bot), # runs our coded bot, and we pass our bot object 
            Computer(Race.Terran, Difficulty.Hard)], # runs a pre-made computer agent, with a hard difficulty.
            realtime=False, # When set to True, the agent is limited in how long each step can take to process.
        )

# This is the environment itself where Step and Reset are defined
class QueueEnv(gym.Env):
    def __init__(self, config=None, render_mode=None): # None, "human", "rgb_array"
        super(QueueEnv, self).__init__()
        self.action_space = Discrete(constants.NUMBER_OF_ACTIONS)
        self.observation_space = MultiDiscrete(constants.OBSERVATION_SPACE_ARRAY)
        self.current_episode_reward = 0 
        self.rewards_file = os.path.join(models_dir, "episode_rewards.csv")  # Define el archivo CSV

    def step(self, action):
        # Send an action to the Bot
        self.gameThread.action_in.put(action)

        # Get the result
        out = self.gameThread.result_out.get()               
        observation = out["observation"].astype(np.int32)
        reward = out["reward"]
        done = out["done"]
        truncated = out["truncated"]
        info = out["info"]

        self.current_episode_reward += reward 

        if done:
            with open(self.rewards_file, 'a', newline='') as file:
                writer = csv.writer(file)
                if file.tell() == 0:
                    writer.writerow(["Total Episode Reward"])
                writer.writerow([self.current_episode_reward])
            self.current_episode_reward = 0 
        observation = np.clip(observation, 0, np.inf).astype(np.int32)
        return observation, reward, done, truncated, info
    
    def reset(self, *, seed=None, options=None):
        print("--- RESETTING ENVIRONMENT ---")
        time.sleep(5)
        observation = constants.EMPTY_OBSERVATION
        info = {}
        self.gameThread = GameThread()
        self.gameThread.start()
        observation = np.clip(observation, 0, np.inf).astype(np.int32)
        return observation, info


def make_env():
    def _init():
        env = QueueEnv()
        return env
    return _init

def train_ppo():
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    num_envs = constants.NUMBER_OF_CONCURRENT_EXECUTIONS
    env = SubprocVecEnv([make_env() for i in range(num_envs)])

    model_path = f"models\{model_name}\model.zip"

    if os.path.exists(model_path):
        print("Loading existing model")
        model = PPO.load(model_path, env=env)
    else:
        print("Creating new model")
        model = PPO('MlpPolicy', env, verbose=1, tensorboard_log="./ppo_tensorboard")

    iters = 0
    while iters < constants.NUMBER_OF_ITERATIONS:
        print("On iteration: ", iters)
        iters += 1
        result = model.learn(total_timesteps=constants.TIMESTEPS, reset_num_timesteps=False, tb_log_name="PPO")
        model.save(f"{models_dir}/model")
    env.close()

if __name__ == "__main__":
    train_ppo()
    