# fast_snake/env.py

import gym
from gym import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from .core import FastSnake, UP, DOWN, LEFT, RIGHT

class FastSnakeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi', 'rgb_array']}
    
    def __init__(self, 
                 width: int = 10, 
                 height: int = 10, 
                 num_apples: int = 5, 
                 max_rounds: int = 100,
                 num_external_snakes: int = 1,
                 num_random_snakes: int = 1):
        """
        Initialize Fast Snake Game Environment.
        
        Args:
            width: Board width
            height: Board height
            num_apples: Number of apples on the board
            max_rounds: Maximum number of rounds before game ends
            num_external_snakes: Number of snakes controlled by the environment
            num_random_snakes: Number of additional random-policy snakes
        """
        super().__init__()
        
        self.width = width
        self.height = height
        self.num_apples = num_apples
        self.max_rounds = max_rounds
        self.num_external_snakes = num_external_snakes
        self.num_random_snakes = num_random_snakes
        
        # Define action spaces
        if num_external_snakes == 1:
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Tuple([spaces.Discrete(4)] * num_external_snakes)
        
        # Define observation spaces - using 3 channels
        # Channel 0: Current snake (1=body, 2=head)
        # Channel 1: Apples (1=apple)
        # Channel 2: Other snakes (1=body, 2=head)
        obs_shape = (3, height, width)
        if num_external_snakes == 1:
            self.observation_space = spaces.Box(
                low=0, high=2,
                shape=obs_shape,
                dtype=np.int8
            )
        else:
            self.observation_space = spaces.Tuple([
                spaces.Box(
                    low=0, high=2,
                    shape=obs_shape,
                    dtype=np.int8
                )
            ] * num_external_snakes)
        
        # Initialize game
        self.game = None
        self.external_snake_ids = []
        self.random_snake_ids = []
        
        # Store last scores for reward calculation
        self.last_scores = {}
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment."""
        super().reset(seed=seed)
        
        # Create new game instance
        self.game = FastSnake(
            width=self.width,
            height=self.height,
            num_apples=self.num_apples,
            max_rounds=self.max_rounds
        )
        
        # Reset snake tracking
        self.external_snake_ids = []
        self.random_snake_ids = []
        self.last_scores = {}
        
        # Add external snakes
        for i in range(self.num_external_snakes):
            snake_id = f"external_{i+1}"
            self.external_snake_ids.append(snake_id)
            self.game.add_snake(snake_id)
            self.last_scores[snake_id] = 0
        
        # Add random snakes
        for i in range(self.num_random_snakes):
            snake_id = f"random_{i+1}"
            self.random_snake_ids.append(snake_id)
            self.game.add_snake(snake_id)
            self.last_scores[snake_id] = 0
        
        return self._get_obs(), self._get_info()
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one environment step."""
        # Convert action(s) to dict format
        if self.num_external_snakes == 1:
            actions = {self.external_snake_ids[0]: action}
        else:
            actions = {
                snake_id: act 
                for snake_id, act in zip(self.external_snake_ids, action)
            }
        
        # Add random actions for random snakes
        for snake_id in self.random_snake_ids:
            if self.game.snakes[snake_id]['alive']:
                actions[snake_id] = np.random.randint(4)
        
        # Execute game step
        observations, rewards, done, info = self.game.step(actions)
        
        # Calculate rewards with additional incentives
        rewards = self._calculate_rewards()
        
        # Extract relevant observation and reward for external snakes
        if self.num_external_snakes == 1:
            obs = observations[self.external_snake_ids[0]]
            reward = rewards[self.external_snake_ids[0]]
        else:
            obs = tuple(observations[sid] for sid in self.external_snake_ids)
            reward = tuple(rewards[sid] for sid in self.external_snake_ids)
        
        # Update last scores
        for snake_id in self.game.snakes:
            self.last_scores[snake_id] = self.game.scores[snake_id]
        
        return obs, reward, done, False, self._get_info()
    
    def _calculate_rewards(self) -> Dict[str, float]:
        """Calculate rewards for all snakes."""
        rewards = {}
        for snake_id in self.game.snakes:
            reward = 0.0
            
            # Reward for eating apple (score increase)
            current_score = self.game.scores[snake_id]
            if current_score > self.last_scores[snake_id]:
                reward += 1.0
            
            # Penalty for dying
            if not self.game.snakes[snake_id]['alive']:
                reward -= 1.0
            
            # Small penalty for each step to encourage efficient paths
            reward -= 0.01
            
            rewards[snake_id] = reward
        
        return rewards
    
    def _get_obs(self) -> np.ndarray:
        """Get observations for external snakes."""
        observations = self.game.get_observations()
        
        if self.num_external_snakes == 1:
            return observations[self.external_snake_ids[0]]
        else:
            return tuple(
                observations[snake_id] 
                for snake_id in self.external_snake_ids
            )
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the current state."""
        return {
            'scores': {
                sid: self.game.scores[sid] 
                for sid in self.external_snake_ids
            },
            'round': self.game.round_number,
            'alive': {
                sid: self.game.snakes[sid]['alive'] 
                for sid in self.external_snake_ids
            },
            'game_over': self.game.game_over,
            'all_scores': self.game.scores
        }
    
    def render(self, mode: str = 'human'):
        """Render the game state."""
        if mode in ['human', 'ansi']:
            return self.game.render_text()
        elif mode == 'rgb_array':
            # TODO: Implement RGB rendering if needed
            raise NotImplementedError("RGB array rendering not implemented yet")
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def game_state_text(self) -> str:
        """Get a text representation of the game state."""
        snake_id = self.external_snake_ids[0]
        snake_head = self.game.snakes[snake_id]['positions'][0]
        apple_positions = self.game.apples
        
        # Create the description
        apples_str = ", ".join(str(a) for a in apple_positions)
        enemy_snakes = [
            (i, sid, self.game.snakes[sid]['positions'][0])
            for i, sid in enumerate(self.game.snakes.keys(), start=1)
            if sid != snake_id and self.game.snakes[sid]['alive']
        ]
        
        enemy_str = "\n".join([
            f"* Snake #{i} is at position {pos}"
            for i, _, pos in enemy_snakes
        ])
        
        return (
            f"The board size is {self.width}x{self.height}. Normal (X,Y) coordinates are used. "
            f"Coordinates range from (0,0) at bottom left to ({self.width-1},{self.height-1}) at top right.\n"
            f"Apples at: {apples_str}\n\n"
            f"Your snake ID: 1 which is currently positioned at {snake_head}\n\n"
            f"Enemy snakes positions:\n{enemy_str}\n\n"
            f"Game state:\n{self.game.render_text()}\n\n"
        )