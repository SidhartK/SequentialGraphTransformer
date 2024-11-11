import gymnasium as gym
from gymnasium import spaces
import numpy as np


class SimpleDiscreteEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    This is a template for creating custom gym environments.
    """
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 30}

    def __init__(self, render_mode=None):
        super(SimpleDiscreteEnv, self).__init__()
        
        # Define action and observation space
        # They must be gym.spaces objects
        
        # Example for continuous action space:
        # self.action_space = spaces.Box(
        #     low=-1.0,
        #     high=1.0,
        #     shape=(1,),
        #     dtype=np.float32
        # )
        
        # Example for discrete action space:
        self.action_space = spaces.Discrete(2)
        
        # Example for continuous observation space:
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(4,),  # Example shape, modify as needed
            dtype=np.float32
        )
        
        # Example for discrete observation space:
        # self.observation_space = spaces.Discrete(8)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Initialize state
        self.state = None
        self.steps = 0
        
        # If using pygame for rendering
        self.window = None
        self.clock = None

    def step(self, action):
        """
        Execute one time step within the environment.
        """
        assert self.state is not None, "Call reset before using step method."
        
        # Execute action
        # Update state based on action
        # Calculate reward
        # Check if episode is done
        
        self.steps += 1
        
        # Example implementation:
        # self.state = np.array([0.0, 0.0, 0.0, 0.0])  # Update with your state
        reward = np.random.normal(action * np.linalg.norm(self.state), 1)  # Calculate your reward
        terminated = (self.steps >= 10)  # Check if episode should end
        truncated = False  # Check if episode was artificially terminated
        info = {}  # Additional info for debugging
        self.state = np.random.multivariate_normal(mean=np.ones(4) * (2 - 2 * action), cov=0.1 * np.eye(4))

        if self.render_mode == "human":
            self._render_frame()

        return self.state, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """
        Reset the environment to initial state.
        """
        super().reset(seed=seed)  # Seed the RNG if provided
        
        # Initialize state
        self.state = np.array([0.0, 0.0, 0.0, 0.0])  # Set your initial state
        self.steps = 0
        
        # Info for reset
        info = {}
        
        if self.render_mode == "human":
            self._render_frame()

        return self.state, info

    def render(self):
        """
        Render the environment.
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def _render_frame(self):
        """
        Render the current frame.
        """
        # Example rendering using pygame:
        """
        if self.window is None and self.render_mode == "human":
            import pygame
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((600, 400))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # Draw the environment state
        # Update the window
        """
        pass

    def close(self):
        """
        Clean up resources.
        """
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None

    def seed(self, seed=None):
        """
        Set the seed for reproducible results.
        """
        super().reset(seed=seed)
        return [seed]

    def get_observation(self):
        """
        Helper method to process and return current observation.
        """
        return self.state

    def get_reward(self, state, action, next_state):
        """
        Helper method to calculate reward.
        """
        return 0.0

    def is_terminal(self, state):
        """
        Helper method to check if state is terminal.
        """
        return False