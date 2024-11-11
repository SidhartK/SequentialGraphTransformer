import numpy as np

def run_simulations(model, env, n, print_every=None):
    random_policy = model is None
    total_rewards = []
    for i in range(n):
        verbose = (print_every is not None) and (i % print_every == 0)
        if verbose:
            print(f"Episode {i}")
        # print("Starting Episode ", i)
        obs, _ = env.reset()
        total_reward = 0
        while True:
            if not random_policy:
                action, _states = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            if verbose:
                print(f"Action: {action}, Obs: {obs}, Reward: {reward}, Done: {terminated or truncated}")
            total_reward += reward
            # env.render()
            if terminated or truncated:
                break
        total_rewards.append(total_reward)
    print(f"Episode rewards: {np.mean(total_rewards)} +/- {np.std(total_rewards)}")

