# fast_snake/core.py

import numpy as np
from typing import List, Tuple, Dict, Optional
from collections import deque

# Constants
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

class FastSnake:
    """Optimized snake game implementation using NumPy arrays."""
    
    def __init__(self, 
                 width: int, 
                 height: int, 
                 max_rounds: int = 100, 
                 num_apples: int = 1, 
                 apple_reward: int = 1,
                 apple_rng=None, 
                 num_bananas: int = 1, 
                 banana_reward: int = 5,
                 banana_rng=None, 
                 num_fires: int = 2, 
                 fire_reward: int = -1, 
                 fire_rng=None, 
                 ):
        # Board representation constants
        self.EMPTY = 100
        self.SNAKE_BODY = 101
        self.APPLE = 102
        self.SNAKE_HEAD = 103
        self.BANANA = 104
        self.FIRE = 105
        
        self.width = width
        self.height = height
        self.max_rounds = max_rounds
        self.num_apples = num_apples if num_apples is not None else 0
        self.apple_reward = apple_reward if apple_reward is not None else 0
        self.num_bananas = num_bananas if num_bananas is not None else 0
        self.banana_reward = banana_reward if banana_reward is not None else 0
        self.num_fires = num_fires if num_fires is not None else 0
        self.fire_reward = fire_reward if fire_reward is not None else 0
        
        # Use provided RNGs or create new ones
        self.apple_rng = apple_rng if apple_rng is not None else np.random.RandomState()
        self.banana_rng = banana_rng if banana_rng is not None else np.random.RandomState()
        self.fire_rng = fire_rng if fire_rng is not None else np.random.RandomState()
        
        # Pre-compute movement deltas for efficiency
        self.MOVE_DELTAS = {
            UP: np.array([0, 1]),
            DOWN: np.array([0, -1]),
            LEFT: np.array([-1, 0]),
            RIGHT: np.array([1, 0])
        }
        
        # Initialize game state
        self.reset()
    
    def reset(self):
        """Reset the game state."""
        self.board = np.full((self.height, self.width), self.EMPTY, dtype=np.int8)
        self.round_number = 0
        self.game_over = False
        
        # Initialize snakes dict with numpy arrays for positions
        self.snakes = {}
        self.scores = {}
        
        # Place initial objects
        self.apples = []
        for _ in range(self.num_apples):
            self._place_apple()
            
        self.bananas = []
        for _ in range(self.num_bananas):
            self._place_banana()
            
        self.fires = []
        for _ in range(self.num_fires):
            self._place_fire()
            
        return self.get_state()
    
    def add_snake(self, snake_id: str) -> None:
        """Add a new snake to the game."""
        if snake_id in self.snakes:
            raise ValueError(f"Snake {snake_id} already exists")
            
        # Initialize snake with random position
        pos = self._random_free_cell()
        self.snakes[snake_id] = {
            'positions': deque([pos]),
            'alive': True
        }
        self.scores[snake_id] = 0
        self._update_board()
    
    def _random_free_cell(self) -> Tuple[int, int]:
        """Find a random empty cell efficiently."""
        empty_cells = np.argwhere(self.board == self.EMPTY)
        if len(empty_cells) == 0:
            raise RuntimeError("No empty cells available")
        
        # Sort empty cells for deterministic selection with same seed
        # This ensures consistent results even if board state changes
        empty_cells = sorted(empty_cells, key=lambda cell: (cell[0], cell[1]))
        
        idx = self.apple_rng.randint(len(empty_cells))
        return tuple(empty_cells[idx])
    
    def _random_free_cell_banana(self) -> Tuple[int, int]:
        """Find a random empty cell for bananas."""
        empty_cells = np.argwhere(self.board == self.EMPTY)
        if len(empty_cells) == 0:
            raise RuntimeError("No empty cells available")
        
        empty_cells = sorted(empty_cells, key=lambda cell: (cell[0], cell[1]))
        idx = self.banana_rng.randint(len(empty_cells))
        return tuple(empty_cells[idx])
    
    def _random_free_cell_fire(self) -> Tuple[int, int]:
        """Find a random empty cell for fires."""
        empty_cells = np.argwhere(self.board == self.EMPTY)
        if len(empty_cells) == 0:
            raise RuntimeError("No empty cells available")
        
        empty_cells = sorted(empty_cells, key=lambda cell: (cell[0], cell[1]))
        idx = self.fire_rng.randint(len(empty_cells))
        return tuple(empty_cells[idx])
    
    def _place_apple(self) -> None:
        """Place a new apple on an empty cell."""
        try:
            pos = self._random_free_cell()
            self.apples.append(pos)
            y, x = pos
            self.board[y, x] = self.APPLE
        except RuntimeError:
            pass  # No empty cells for apple
    
    def _place_banana(self) -> None:
        """Place a new banana on an empty cell."""
        try:
            pos = self._random_free_cell_banana()
            self.bananas.append(pos)
            y, x = pos
            self.board[y, x] = self.BANANA
        except RuntimeError:
            pass  # No empty cells for banana
    
    def _place_fire(self) -> None:
        """Place a new fire on an empty cell."""
        try:
            pos = self._random_free_cell_fire()
            self.fires.append(pos)
            y, x = pos
            self.board[y, x] = self.FIRE
        except RuntimeError:
            pass  # No empty cells for fire
    
    def _update_board(self) -> None:
        """Update the board state efficiently."""
        # Clear the board
        self.board.fill(self.EMPTY)
        
        # Place apples
        for ax, ay in self.apples:
            self.board[ay, ax] = self.APPLE
            
        # Place bananas
        for bx, by in self.bananas:
            self.board[by, bx] = self.BANANA
            
        # Place fires
        for fx, fy in self.fires:
            self.board[fy, fx] = self.FIRE
        
        # Place snakes
        for snake_id, snake in self.snakes.items():
            if not snake['alive']:
                continue
                
            # Place body
            for x, y in list(snake['positions'])[1:]:
                self.board[y, x] = self.SNAKE_BODY
                
            # Place head
            positions_list = list(snake['positions'])
            if positions_list: # Check if snake has any positions
                hx, hy = positions_list[0]
                if 0 <= hy < self.height and 0 <= hx < self.width: # Check bounds
                    self.board[hy, hx] = self.SNAKE_HEAD
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, dict]:
        """
        Execute one game step with given actions.
        
        Args:
            actions: Dict mapping snake_id to action (0=UP, 1=DOWN, 2=LEFT, 3=RIGHT)
            
        Returns:
            observations: Dict mapping snake_id to observation array
            rewards: Dict mapping snake_id to reward value
            done: Whether the game is over
            info: Additional information
        """
        if self.game_over:
            raise RuntimeError("Game is already over")
            
        rewards = {sid: 0.0 for sid in self.snakes}
        
        # ---- PHASE 1: Calculate all new head positions ----
        new_heads = {}
        for snake_id, action in actions.items():
            if not self.snakes[snake_id]['alive']:
                continue
                
            head = np.array(self.snakes[snake_id]['positions'][0])
            new_heads[snake_id] = tuple(head + self.MOVE_DELTAS[action])
        
        # ---- PHASE 2: Check for head-to-head collisions ----
        # Find duplicate head positions (head-to-head collisions)
        head_counts = {}
        for snake_id, new_head in new_heads.items():
            if new_head in head_counts:
                head_counts[new_head].append(snake_id)
            else:
                head_counts[new_head] = [snake_id]
        
        # Mark snakes involved in head-to-head collisions as not alive
        for pos, snake_ids in head_counts.items():
            if len(snake_ids) > 1:
                # More than one snake moving to the same position - collision!
                for snake_id in snake_ids:
                    if self.snakes[snake_id]['alive']:
                        self.snakes[snake_id]['alive'] = False
                        rewards[snake_id] -= 1.0
                        
        # ---- PHASE 3: Process individual snake moves and collisions ----
        for snake_id, new_head in new_heads.items():
            snake = self.snakes[snake_id]
            if not snake['alive']:
                continue  # Skip already dead snakes (from head-to-head collisions)
                
            x, y = new_head
            
            # Check bounds
            if not (0 <= x < self.width and 0 <= y < self.height):
                snake['alive'] = False
                rewards[snake_id] -= 1.0
                continue
            
            # Check collisions with snake bodies on the current board
            # (This won't catch head-to-head collisions, which we handled above)
            if self.board[y, x] in [self.SNAKE_BODY, self.SNAKE_HEAD]:
                snake['alive'] = False
                rewards[snake_id] -= 1.0
                continue
            
            # Move snake
            snake['positions'].appendleft(new_head)
            
            # Check apple
            if new_head in self.apples:
                self.scores[snake_id] += self.apple_reward
                rewards[snake_id] += float(self.apple_reward)
                self.apples.remove(new_head)
                self._place_apple()
            # Check banana
            elif new_head in self.bananas:
                self.scores[snake_id] += self.banana_reward
                rewards[snake_id] += float(self.banana_reward)
                self.bananas.remove(new_head)
                self._place_banana()
            # Check fire
            elif new_head in self.fires:
                self.scores[snake_id] += self.fire_reward
                rewards[snake_id] += float(self.fire_reward)
                self.fires.remove(new_head)
                self._place_fire()
            else:
                snake['positions'].pop()
        
        # Update the board after all snakes have moved
        self._update_board()
        self.round_number += 1
        
        # Check game over conditions
        alive_snakes = sum(s['alive'] for s in self.snakes.values())
        self.game_over = (alive_snakes <= 1) or (self.round_number >= self.max_rounds)
        
        return (
            self.get_observations(),
            rewards,
            self.game_over,
            {'scores': self.scores.copy(), 'round': self.round_number}
        )
    
    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get observations for all snakes."""
        return {
            snake_id: self._get_snake_observation(snake_id)
            for snake_id in self.snakes
        }
    
    def _get_snake_observation(self, snake_id: str) -> np.ndarray:
        """Get observation from one snake's perspective."""
        obs = np.zeros((5, self.height, self.width), dtype=np.int8)
        
        # Channel 0: Current snake
        if self.snakes[snake_id]['alive']:
            positions = self.snakes[snake_id]['positions']
            head = positions[0]
            # Mark head
            obs[0, head[1], head[0]] = 2
            # Mark body
            for x, y in list(positions)[1:]:
                obs[0, y, x] = 1
        
        # Channel 1: Apples
        for x, y in self.apples:
            obs[1, y, x] = 1
            
        # Channel 2: Bananas
        for x, y in self.bananas:
            obs[2, y, x] = 1
            
        # Channel 3: Fires
        for x, y in self.fires:
            obs[3, y, x] = 1
        
        # Channel 4: Other snakes
        for other_id, snake in self.snakes.items():
            if other_id != snake_id and snake['alive']:
                positions = snake['positions']
                head = positions[0]
                # Mark head
                obs[4, head[1], head[0]] = 2
                # Mark body
                for x, y in list(positions)[1:]:
                    obs[4, y, x] = 1
        
        return obs
    
    def get_state(self) -> np.ndarray:
        """Get the raw board state."""
        return self.board.copy()
    
    def render_text(self) -> str:
        """
        Returns a string representation of the board with:
        # = empty space
        A = apple
        B = banana
        F = fire
        T = snake tail
        1,2,3... = snake head (showing player number based on order in self.snakes)
        Now with (0,0) at bottom left and x-axis labels at bottom
        """
        # Create empty board
        board = [['#' for _ in range(self.width)] for _ in range(self.height)]

        # Place apples
        for ax, ay in self.apples:
            if 0 <= ay < self.height and 0 <= ax < self.width:
                board[ay][ax] = 'A'
                
        # Place bananas
        for bx, by in self.bananas:
            if 0 <= by < self.height and 0 <= bx < self.width:
                board[by][bx] = 'B'
                
        # Place fires
        for fx, fy in self.fires:
            if 0 <= fy < self.height and 0 <= fx < self.width:
                board[fy][fx] = 'F'

        # Place snakes
        for i, (snake_id, snake_data) in enumerate(self.snakes.items(), start=1):
            if not snake_data['alive']:
                continue

            positions = snake_data['positions']
            # Place snake body
            for pos_idx, (x, y) in enumerate(positions):
                if 0 <= y < self.height and 0 <= x < self.width:
                    if pos_idx == 0:  # Head
                        board[y][x] = str(i)  # Use snake number (1, 2, 3...) for head
                    else:  # Body/tail
                        board[y][x] = 'T'

        # Build the string representation
        result = []
        # Print rows in reverse order (bottom to top)
        for y in range(self.height - 1, -1, -1):
            result.append(f"{y:2d} {' '.join(board[y])}")

        # Add x-axis labels at the bottom
        result.append("   " + " ".join(str(i) for i in range(self.width)))

        return "\n".join(result)