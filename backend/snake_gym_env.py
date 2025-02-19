import gym
from gym import spaces
import numpy as np
from typing import Optional, Dict, Any, Tuple, List
from .main import UP, DOWN, LEFT, RIGHT, RandomPlayer, SnakeGame

class SnakeGameEnv(gym.Env):
    metadata = {'render_modes': ['human', 'ansi', 'rgb_array']}
    
    def __init__(self, 
                 width: int = 10, 
                 height: int = 10, 
                 num_apples: int = 5, 
                 max_rounds: int = 100,
                 num_external_snakes: int = 1,
                 num_random_snakes: int = 1):
        """
        Initialize Snake Game Environment.
        
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
        
        # Define action space for each controlled snake (UP=0, DOWN=1, LEFT=2, RIGHT=3)
        if num_external_snakes == 1:
            self.action_space = spaces.Discrete(4)
        else:
            self.action_space = spaces.Tuple([spaces.Discrete(4)] * num_external_snakes)
        
        self.action_to_direction = {
            0: UP,
            1: DOWN,
            2: LEFT,
            3: RIGHT
        }
        
        # Define observation space
        # We'll use a 3-channel grid where:
        # Channel 0: Controlled snakes (1 for body, 2 for head)
        # Channel 1: Apples (1 for apple)
        # Channel 2: Other snakes (1 for body, 2 for head)
        if num_external_snakes == 1:
            self.observation_space = spaces.Box(
                low=0,
                high=2,
                shape=(3, height, width),
                dtype=np.int8
            )
        else:
            self.observation_space = spaces.Tuple([
                spaces.Box(
                    low=0,
                    high=2,
                    shape=(3, height, width),
                    dtype=np.int8
                )
            ] * num_external_snakes)
        
        # Initialize game
        self.game = None
        self.external_snake_ids = []
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        
        # Create new game instance
        self.game = SnakeGame(
            width=self.width,
            height=self.height,
            max_rounds=self.max_rounds,
            num_apples=self.num_apples
        )
        
        # Add externally controlled snakes
        self.external_snake_ids = []
        for i in range(self.num_external_snakes):
            snake_id = f"external_{i+1}"
            self.external_snake_ids.append(snake_id)
            self.game.add_snake(snake_id, player=None, is_external=True)
        
        # Add random-policy snakes
        for i in range(self.num_random_snakes):
            snake_id = f"random_{i+1}"
            self.game.add_snake(snake_id, player=RandomPlayer(snake_id), is_external=False)
        
        # Record initial state
        self.game.record_history()
        
        return self._get_obs(), self._get_info()
        
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        # Convert action(s) to direction(s)
        if self.num_external_snakes == 1:
            actions = [action]
        else:
            actions = action
            
        external_moves = {
            snake_id: self.action_to_direction[act]
            for snake_id, act in zip(self.external_snake_ids, actions)
        }
        
        # Run a round of the game with our moves
        self.game.run_round(external_moves=external_moves)
        
        # Get observation and info
        observation = self._get_obs()
        info = self._get_info()
        
        # Calculate reward - now returns individual rewards per snake
        reward = self._calculate_reward()
        
        # Check if game is done
        terminated = self.game.game_over
        truncated = False  # We don't truncate episodes early
        
        return observation, reward, terminated, truncated, info
    
    def _calculate_reward(self) -> List[float]:
        """Calculate rewards for each externally controlled snake.
        
        Returns:
            List[float]: List of rewards, one for each external snake in order
        """
        rewards = []
        
        for snake_id in self.external_snake_ids:
            snake_reward = 0.0
            
            # Reward for eating apple (score increase)
            current_score = self.game.scores[snake_id]
            last_score = getattr(self, f'_last_score_{snake_id}', 0)
            if current_score > last_score:
                snake_reward += 1.0
            setattr(self, f'_last_score_{snake_id}', current_score)
            
            # Penalty for dying
            if not self.game.snakes[snake_id].alive:
                snake_reward -= 2.0
            
            # Small penalty for each step to encourage efficient paths
            snake_reward -= 0.01
            
            rewards.append(snake_reward)
        
        rewards = None if len(rewards) == 0 else rewards[0] if len(rewards) == 1 else rewards
        
        return rewards
    
    def game_state_text(self):
            game_state = self.game.get_current_state()
            apples_str = ", ".join(str(a) for a in game_state.apples)
            snake_id = self.external_snake_ids[0] 
            snake_head_num = 1
            return (
                f"The board size is {game_state.width}x{game_state.height}. Normal (X, Y) coordinates are used. Coordinates range from (0, 0) at bottom left to ({game_state.width-1}, {game_state.height-1}) at top right.\n"
                f"Apples at: {apples_str}\n\n"
                f"Your snake ID: {snake_head_num} which is currently positioned at {game_state.snake_positions[snake_id][0]} with body at {game_state.snake_positions[snake_id][1:]}\n\n"
                f"Enemy snakes positions:\n" + "\n".join([f"* Snake #{i} is at position {pos[0]} with body at {pos[1:]}" for i, (sid, pos) in enumerate(game_state.snake_positions.items(), start=1) if sid != snake_id]) + "\n\n"
                f"Game state:\n"
                f"{game_state.print_board()}\n\n"
                # f"--Your last move information:--\n\n"
                # f"Direction: {self.move_history[-1][self.snake_id]['direction'] if self.move_history else 'None'}\n"
                # f"Rationale: {self.move_history[-1][self.snake_id]['rationale'] if self.move_history else 'None'}\n\n"
                # f"--End of your last move information.--\n\n"
            )
    
    def render(self, mode: str = 'human'):
        if mode == 'human' or mode == 'ansi':
            return self.game.get_current_state().print_board()
        elif mode == 'rgb_array':
            # TODO: Implement RGB rendering if needed
            raise NotImplementedError("RGB array rendering not implemented yet")
        else:
            raise ValueError(f"Unsupported render mode: {mode}")
    
    def _get_obs(self) -> np.ndarray:
        """Convert game state to observation array(s)."""
        if self.num_external_snakes == 1:
            return self._get_single_snake_obs(self.external_snake_ids[0])
        else:
            return tuple(self._get_single_snake_obs(snake_id) 
                        for snake_id in self.external_snake_ids)
    
    def _get_single_snake_obs(self, snake_id: str) -> np.ndarray:
        """Get observation array from perspective of one snake."""
        obs = np.zeros((3, self.height, self.width), dtype=np.int8)
        state = self.game.get_current_state()
        
        # Channel 0: Current snake
        if snake_id in state.snake_positions:
            positions = state.snake_positions[snake_id]
            if positions:  # If snake is alive
                # Mark body
                for x, y in positions[1:]:
                    obs[0, y, x] = 1
                # Mark head
                head_x, head_y = positions[0]
                obs[0, head_y, head_x] = 2
        
        # Channel 1: Apples
        for x, y in state.apples:
            obs[1, y, x] = 1
        
        # Channel 2: Other snakes
        for other_id, positions in state.snake_positions.items():
            if other_id != snake_id and positions:  # If it's another snake and it's alive
                # Mark body
                for x, y in positions[1:]:
                    obs[2, y, x] = 1
                # Mark head
                head_x, head_y = positions[0]
                obs[2, head_y, head_x] = 2
        
        return obs
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional info about the current state."""
        state = self.game.get_current_state()
        return {
            'scores': {sid: self.game.scores[sid] for sid in self.external_snake_ids},
            'round': state.round_number,
            'alive': {sid: self.game.snakes[sid].alive for sid in self.external_snake_ids},
            'game_over': self.game.game_over,
            'all_scores': self.game.scores
        }
