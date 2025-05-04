import gym
from gym import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from .core import FastSnake, UP, DOWN, LEFT, RIGHT
import random # Import random

class FastSnakeEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi', 'rgb_array']}
    
    STRING_ACTION_MAP = {
        "up": 0,
        "down": 1,
        "left": 2,
        "right": 3
    }

    def __init__(self, 
                 width: int = 10, 
                 height: int = 10, 
                 max_rounds: int = 100,
                 num_external_snakes: int = 1,
                 num_random_snakes: int = 1,
                 death_reward: int = -2,
                 step_reward: int = -0.01,
                 num_apples: int = 5, 
                 apple_reward: int = 1,
                 num_bananas: int = None,
                 banana_reward: int = None,
                 num_fires: int = None, 
                 fire_reward: int = None,
                 hill_direction: str = None,
                 destroy_at_bottom: bool = False,
                 include_absent_objects: bool = None):
        """
        Initialize Fast Snake Game Environment.
        
        Args:
            width: Board width
            height: Board height
            num_apples: Number of apples on the board
            max_rounds: Maximum number of rounds before game ends
            num_external_snakes: Number of snakes controlled by the environment
            num_random_snakes: Number of additional random-policy snakes
            num_bananas: Number of bananas on the board (worth more points)
            banana_reward: Points awarded for eating a banana
            num_fires: Number of fires on the board (worth negative points)
            fire_reward: Points deducted for walking on fire
            apple_reward: Points awarded for eating an apple
            hill_direction: Direction of the hill the apples will roll down(up, down, left, right)
            include_absent_objects: Whether to include absent objects in the observation
        """
        super().__init__()
        
        self.width = width
        self.height = height
        self.max_rounds = max_rounds
        self.num_external_snakes = num_external_snakes
        self.num_random_snakes = num_random_snakes
        self.death_reward = death_reward
        self.step_reward = step_reward
        self.num_apples = num_apples
        self.apple_reward = apple_reward
        self.num_bananas = num_bananas
        self.banana_reward = banana_reward
        self.num_fires = num_fires
        self.fire_reward = fire_reward
        self.hill_direction = hill_direction
        self.destroy_at_bottom = destroy_at_bottom
        self.include_absent_objects = include_absent_objects
        # Validate hill_direction with fires
        if hill_direction is not None and num_fires is not None and num_fires > 0:
            raise ValueError("hill_direction is not compatible with fires. Please set num_fires to 0 or hill_direction to None.")
        
        # Define action spaces
        if num_external_snakes == 1:
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Tuple([spaces.Discrete(4)] * num_external_snakes)
        
        # Define observation spaces with 5 channels
        # Channel 0: Current snake's head location (binary)
        # Channel 1: All snake bodies (including heads) (binary)
        # Channel 2: Apples (binary)
        # Channel 3: Bananas (binary)
        # Channel 4: Fires (binary)

        # If include_absent_objects is False, the observation space will be 2 + which of apples, bananas, and fires are present
        if not include_absent_objects:
            obs_shape = (2 + (num_apples > 0) + (num_bananas > 0) + (num_fires > 0), height, width)
        else:
            obs_shape = (5, height, width)
        if num_external_snakes == 1:
            self.observation_space = spaces.Box(
                low=0, high=1,  # Binary values only
                shape=obs_shape,
                dtype=np.int8
            )
        else:
            self.observation_space = spaces.Tuple([
                spaces.Box(
                    low=0, high=1,  # Binary values only
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
        
        # Create separate RNGs for different purposes
        if seed is not None:
            # Create a separate RNG for snake actions - with a derived seed
            self.snake_rng = np.random.RandomState(seed + 1)
            # Create a separate RNG for apple placement - with a different derived seed
            apple_rng = np.random.RandomState(seed + 2)
            # Create RNGs for banana and fire placement
            banana_rng = np.random.RandomState(seed + 3)
            fire_rng = np.random.RandomState(seed + 4)
        else:
            self.snake_rng = None
            apple_rng = None
            banana_rng = None
            fire_rng = None
        
        # Create new game instance with the separate RNGs
        self.game = FastSnake(
            width=self.width,
            height=self.height,
            max_rounds=self.max_rounds,
            num_apples=self.num_apples,
            apple_reward=self.apple_reward,
            apple_rng=apple_rng,  # RNG for apple placement
            num_bananas=self.num_bananas,
            banana_reward=self.banana_reward,
            banana_rng=banana_rng,
            num_fires=self.num_fires,
            fire_reward=self.fire_reward,
            fire_rng=fire_rng,
            hill_direction=self.hill_direction,  # Pass the hill direction to core
            destroy_at_bottom=self.destroy_at_bottom,
            include_absent_objects=self.include_absent_objects        
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
        """Execute one environment step.
        Returns:
            obs: Observation
            reward: Reward
            done: Whether the episode is over
            truncated: Whether the episode is truncated
            info: Additional info
        """
        # Convert action(s) to dict format
        if self.num_external_snakes == 1:
            actions = {self.external_snake_ids[0]: action}
        else:
            actions = {
                snake_id: act
                for snake_id, act in zip(self.external_snake_ids, action)
            }

        # Add actions for random snakes (matching RandomPlayer logic)
        for snake_id in self.random_snake_ids:
            if self.game.snakes[snake_id]['alive']:
                # actions[snake_id] = np.random.randint(4) # Old truly random logic

                # New logic: Choose random valid move (avoid walls/self)
                snake_data = self.game.snakes[snake_id]
                positions = snake_data['positions']
                head_x, head_y = positions[0]
                possible_actions = { # Map action const to potential new head pos
                    UP:    (head_x, head_y + 1),
                    DOWN:  (head_x, head_y - 1),
                    LEFT:  (head_x - 1, head_y),
                    RIGHT: (head_x + 1, head_y)
                }

                valid_actions = []
                # Convert deque to list once for slicing
                positions_list = list(positions)
                # Only exclude tail if snake has more than 2 segments (head + body)
                body_to_check = positions_list[:-1] if len(positions_list) > 2 else positions_list

                for action, (new_x, new_y) in possible_actions.items():
                    # Check wall collisions
                    if not (0 <= new_x < self.width and 0 <= new_y < self.height):
                        # print(f"RandomPlayer would hit a wall at {new_x}, {new_y}")
                        continue

                    # Check self collisions (excluding tail)
                    if (new_x, new_y) in body_to_check:
                        # print(f"RandomPlayer would hit itself at {new_x}, {new_y}")
                        continue

                    # Check collisions with other snakes (Optional - RandomPlayer doesn't do this)
                    # occupied_by_others = False
                    # for other_id, other_snake in self.game.snakes.items():
                    #     if other_id != snake_id and other_snake['alive']:
                    #         if (new_x, new_y) in other_snake['positions']:
                    #             occupied_by_others = True
                    #             break
                    # if occupied_by_others:
                    #     continue

                    valid_actions.append(action)

                # Choose action
                if valid_actions:
                    # Sort valid_actions for deterministic selection with same seed
                    valid_actions.sort()
                    # Use snake_rng if available, otherwise use standard random
                    if self.snake_rng is not None:
                        chosen_action = self.snake_rng.choice(valid_actions)
                    else:
                        chosen_action = random.choice(valid_actions)
                else:
                    # Trapped, choose a random action (likely dying)
                    # print(f"RandomPlayer {snake_id} is trapped, choosing a random action")
                    possible_action_keys = list(possible_actions.keys())
                    possible_action_keys.sort()  # Sort for deterministic selection
                    # Use snake_rng if available, otherwise use standard random
                    if self.snake_rng is not None:
                        chosen_action = self.snake_rng.choice(possible_action_keys)
                    else:
                        chosen_action = random.choice(possible_action_keys)

                actions[snake_id] = chosen_action

        # Execute game step
        observations, rewards, done, info = self.game.step(actions)
        
        # Calculate rewards with additional incentives
        for snake_id in self.game.snakes:
            # Penalty for dying
            if not self.game.snakes[snake_id]['alive']:
                rewards[snake_id] += self.death_reward
            
            # Penalty for each step to encourage efficient paths
            rewards[snake_id] += self.step_reward
        
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

    def _get_obs(self) -> np.ndarray:
        """Get observations for external snakes."""
        observations = self.game.get_observations(self.include_absent_objects)
        
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
        
    def env_state_text(self) -> str:
        """Compatibility"""
        return self.game_state_text()
    
    def game_state_text(self) -> str:
        """Get a text representation of the game state, matching SnakeGameEnv format."""
        if not self.game or not self.external_snake_ids:
            return "Game not initialized or no external snake."

        your_snake_id = self.external_snake_ids[0]

        # Create mapping from snake_id to display number (1, 2, ...)
        snake_id_to_number = {sid: i for i, sid in enumerate(self.game.snakes.keys(), start=1)}
        your_snake_number = snake_id_to_number.get(your_snake_id, '?') # Should always find it

        # Ensure your snake exists in the game data
        if your_snake_id not in self.game.snakes or not self.game.snakes[your_snake_id]['alive']:
            your_snake_head_str = "(Dead)"
            your_snake_body_str = "[]"
        else:
            your_snake_positions = list(self.game.snakes[your_snake_id]['positions'])
            your_snake_head = your_snake_positions[0]
            your_snake_body = your_snake_positions[1:]
            your_snake_head_str = str(your_snake_head)
            your_snake_body_str = str(your_snake_body)

        # Get apple positions
        apple_positions = self.game.apples
        apples_str = ", ".join(str(a) for a in apple_positions)
        
        # Get banana positions
        banana_positions = self.game.bananas
        bananas_str = ", ".join(str(b) for b in banana_positions)
        
        # Get fire positions
        fire_positions = self.game.fires
        fires_str = ", ".join(str(f) for f in fire_positions)

        # Get enemy snake positions
        enemy_strs = []
        for sid, snake_data in self.game.snakes.items():
            if sid != your_snake_id and snake_data['alive']:
                enemy_number = snake_id_to_number[sid]
                positions = list(snake_data['positions'])
                head_pos = positions[0]
                body_pos = positions[1:]
                enemy_strs.append(f"* Snake #{enemy_number} has head at position {head_pos} and body segments at {body_pos}")
        enemy_str = "\n".join(enemy_strs)

        # Get board representation using the updated render_text
        board_str = self.game.render_text()
        
        # Include hill direction information if applicable
        hill_direction_str = f"Apples roll {self.hill_direction}" if self.hill_direction else "Apples don't roll"

        # Construct the final string
        return (
            f"The board size is {self.width}x{self.height}. Normal (X, Y) coordinates are used. Coordinates range from (0, 0) at bottom left to ({self.width-1}, {self.height-1}) at top right.\n"
            f"Apples at: {apples_str} (worth {self.apple_reward} points each)\n"
            f"Bananas at: {bananas_str} (worth {self.banana_reward} points each)\n"
            f"Fires at: {fires_str} (worth {self.fire_reward} points each)\n"
            f"{hill_direction_str}\n\n"
            f"Your snake ID: {your_snake_number} with head positioned at {your_snake_head_str} and body segments at {your_snake_body_str}\n\n"
            f"Enemy snakes positions:\n{enemy_str}\n\n"
            f"Game state:\n"
            f"{board_str}\n\n"
        )