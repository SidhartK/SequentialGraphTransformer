import gymnasium as gym
from sb3_contrib import QRDQN

import os
from common import run_simulations

env = gym.make("CartPole-v1")

artifact_dir = "artifacts/cartpole/qrdqn"
os.makedirs(artifact_dir, exist_ok=True)

policy_kwargs = dict(n_quantiles=50)
model = QRDQN("MlpPolicy", env, policy_kwargs=policy_kwargs, learning_rate=2e-4, learning_starts=1_000, verbose=1)
model.learn(total_timesteps=20_000, log_interval=4)
model.save(artifact_dir)


del model # remove to demonstrate saving and loading
model = QRDQN.load(artifact_dir)

run_simulations(None, env)
run_simulations(model, env)


        
        
