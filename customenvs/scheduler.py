import gymnasium as gym
import numpy as np

class SchedulerEnv(gym.Env):
    def __init__(self, k=3, completion_rates=None, rewards=None, max_timesteps=100):
        super(SchedulerEnv, self).__init__()
        
        # Number of task types
        self.k = k
        
        # Set completion rates (c_i) and rewards (R_i) for each task type
        if completion_rates is None:
            completion_rates = np.random.rand(self.k) * 0.2  # Random rates between 0 and 0.2
        if rewards is None:
            rewards = np.random.rand(self.k) * 10  # Random rewards between 0 and 10
        
        # Sort completion rates by decreasing order of rewards
        sorted_indices = np.argsort(rewards)[::-1]  # Sort rewards in descending order
        self.rewards = np.array(rewards)[sorted_indices]
        self.completion_rates = np.array(completion_rates)[sorted_indices]
        
        
        # Task completion threshold
        # self.threshold = 1e-3
        self.operation_cost = 0.1
        self.max_logit = 10

        # State space: remaining work (k values between 0 and 1) + logits for worker distribution (k values)
        self.observation_space = gym.spaces.Box(low=-self.max_logit, high=self.max_logit, shape=(2 * self.k,), dtype=np.float32)
        
        # Action space: update to logits (k-dimensional vector)
        self.action_space = gym.spaces.Box(low=-2*self.max_logit, high=2*self.max_logit, shape=(self.k,), dtype=np.float32)
        
        # Initialize environment variables
        self.max_timesteps = max_timesteps
        self.reset()

    def _update_logits(self, update):
        # self.logits += (update - np.mean(self.logits))
        self.logits = np.clip(self.logits + update, -self.max_logit, self.max_logit)
        

    def reset(self, seed=None, options=None):
        # Initialize remaining work for each task type (between 0 and 1)
        super().reset(seed=seed)  # Seed the RNG if provided
        self.work_remaining = np.ones(self.k)
        
        # Initialize logits for worker distribution over tasks
        # self.logits = np.zeros(self.k)
        self.logits = np.zeros(self.k)
        self._update_logits(np.random.uniform(-self.max_logit, self.max_logit))
        
        
        # Reset timestep
        self.timesteps = 0

        info = {}
        
        if self.render_mode == "human":
            self.render()

        return self._get_obs(), info

    def _get_obs(self):
        # Concatenate work_remaining and logits as the state
        return np.concatenate([self.work_remaining, self.logits])
    
    def _softmax(self, arr):
        return np.exp(arr - np.max(arr)) / np.sum(np.exp(arr - np.max(arr)))
    
    def _state_value(self):
        return -np.dot(self.work_remaining, self.rewards)

    def step(self, action):
        # Ensure action is k-dimensional
        assert action.shape == (self.k,)

        # Update logits with the action and normalize to keep the sum at zero
        self._update_logits(action)
        
        # Convert logits to softmax probabilities for worker distribution
        worker_distribution = self._softmax(self.logits)
        
        # Calculate amount of work completed for each task type this timestep
        work_done = worker_distribution * self.completion_rates
        
        # Update work remaining for each task type
        prev_state_value = self._state_value()
        self.work_remaining = np.maximum(self.work_remaining - work_done, 0.0)
        reward = self._state_value() - prev_state_value - self.operation_cost
        # self.work_remaining[self.work_remaining <= self.threshold] = 0
        
        # Calculate reward based on completed tasks
        # reward = np.sum(self.rewards[(updated_work_remaining <= 0.0) & (self.work_remaining > 0.0)]) - self.operation_cost

        # self.work_remaining = updated_work_remaining

        # for i in range(self.k):
        #     if (self.work_remaining[i] == 0.0) and work_done[i] > 0:
        #         reward += self.rewards[i]
        
        # Check if episode is done (all tasks are completed)
        terminated = np.all(self.work_remaining <= 0.0)
        
        # Increment timestep
        self.timesteps += 1
        truncated = (self.timesteps >= self.max_timesteps)

        if self.render_mode == "human":
            self.render()
        
        # Return updated state, reward, done flag, and info
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self, mode="human"):
        print(f"T: {self.timesteps} -- Work Remaining: {np.round(self.work_remaining, 3)}; Worker Distribution: {np.round(self._softmax(self.logits), 3)}; Completion Rates: {np.round(self.completion_rates, 3)}, Rewards: {self.rewards}")
