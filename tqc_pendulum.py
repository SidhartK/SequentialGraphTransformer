import gymnasium as gym

from sb3_contrib import TQC

from common import run_simulations

import os

import argparse


env = gym.make("Pendulum-v1")
policy_kwargs = dict(n_critics=2, n_quantiles=25)
model = TQC("MlpPolicy", env, top_quantiles_to_drop_per_net=2, verbose=1, policy_kwargs=policy_kwargs)
artifact_dir = "artifacts/scheduler/tqc"
os.makedirs(os.path.dirname(artifact_dir), exist_ok=True)

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Script with train and test flags.")
    
    # Add --train and --test flags
    parser.add_argument("--train", action="store_true", help="Run training mode")
    parser.add_argument("--test", action="store_true", help="Run testing mode")

    parser.add_argument("--num_episodes", type=int, default=100, help="Number of test episodes (default: 100)")
    parser.add_argument("--render", action="store_true", help="Render the test episodes")
    
    # Parse the arguments
    args = parser.parse_args()

    if args.train:
        model.learn(total_timesteps=5_000, log_interval=4)
        model.save(artifact_dir)
        del model

    if args.test:
        model = TQC.load(artifact_dir)
        if args.render:
            args.num_episodes = min(args.num_episodes, 5)
            env = gym.make("Pendulum-v1", render_mode='human')
        
        print(f"Running {args.num_episodes} random episodes")
        run_simulations(None, env, n=args.num_episodes)
        print(f"Running {args.num_episodes} policy episodes")
        run_simulations(model, env, n=args.num_episodes)