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
    
    def __init__(self, width: int, height: int, num_apples: int = 1, max_rounds: int = 100):
        # Board representation constants
        self.EMPTY = 0
        self.SNAKE_BODY = 1
        self.APPLE = 2
        self.SNAKE_HEAD = 3
        
        self.width = width
        self.height = height
        self.num_apples = num_apples
        self.max_rounds = max_rounds
        
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
        self.board = np.zeros((self.height, self.width), dtype=np.int8)
        self.round_number = 0
        self.game_over = False
        
        # Initialize snakes dict with numpy arrays for positions
        self.snakes = {}
        self.scores = {}
        
        # Place initial apples
        self.apples = []
        for _ in range(self.num_apples):
            self._place_apple()
            
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
        idx = np.random.randint(len(empty_cells))
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
    
    def _update_board(self) -> None:
        """Update the board state efficiently."""
        # Clear the board
        self.board.fill(self.EMPTY)
        
        # Place apples
        for ax, ay in self.apples:
            self.board[ay, ax] = self.APPLE
        
        # Place snakes
        for snake_id, snake in self.snakes.items():
            if not snake['alive']:
                continue
                
            # Place body
            for x, y in list(snake['positions'])[1:]:
                self.board[y, x] = self.SNAKE_BODY
                
            # Place head
            hx, hy = snake['positions'][0]
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
        
        # Calculate new heads
        new_heads = {}
        for snake_id, action in actions.items():
            if not self.snakes[snake_id]['alive']:
                continue
                
            head = np.array(self.snakes[snake_id]['positions'][0])
            new_heads[snake_id] = tuple(head + self.MOVE_DELTAS[action])
        
        # Check collisions and update snakes
        for snake_id, new_head in new_heads.items():
            snake = self.snakes[snake_id]
            if not snake['alive']:
                continue
                
            x, y = new_head
            
            # Check bounds
            if not (0 <= x < self.width and 0 <= y < self.height):
                snake['alive'] = False
                rewards[snake_id] -= 1.0
                continue
            
            # Check collisions with snakes
            if self.board[y, x] in [self.SNAKE_BODY, self.SNAKE_HEAD]:
                snake['alive'] = False
                rewards[snake_id] -= 1.0
                continue
            
            # Move snake
            snake['positions'].appendleft(new_head)
            
            # Check apple
            if new_head in self.apples:
                self.scores[snake_id] += 1
                rewards[snake_id] += 1.0
                self.apples.remove(new_head)
                self._place_apple()
            else:
                snake['positions'].pop()
        
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
        obs = np.zeros((3, self.height, self.width), dtype=np.int8)
        
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
        
        # Channel 2: Other snakes
        for other_id, snake in self.snakes.items():
            if other_id != snake_id and snake['alive']:
                positions = snake['positions']
                head = positions[0]
                # Mark head
                obs[2, head[1], head[0]] = 2
                # Mark body
                for x, y in list(positions)[1:]:
                    obs[2, y, x] = 1
        
        return obs
    
    def get_state(self) -> np.ndarray:
        """Get the raw board state."""
        return self.board.copy()
    
    def render_text(self) -> str:
        """Render the game state as text."""
        chars = {
            self.EMPTY: '.',
            self.SNAKE_BODY: 'o',
            self.SNAKE_HEAD: 'O',
            self.APPLE: 'A'
        }
        
        lines = []
        for y in range(self.height-1, -1, -1):
            line = [chars[cell] for cell in self.board[y]]
            lines.append(f"{y:2d} {' '.join(line)}")
        
        # Add x-axis labels
        lines.append("   " + " ".join(str(x) for x in range(self.width)))
        return "\n".join(lines)