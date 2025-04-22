# test_gamestate_text.py

# Adjusted imports for running from modules/SnakeBench directory
from backend.env import FastSnakeEnv
from backend.snake_gym_env import SnakeGameEnv
import numpy as np # Import numpy for coordinate manipulation

# Define common parameters
WIDTH = 10
HEIGHT = 10
NUM_APPLES = 3
MAX_ROUNDS = 10
NUM_EXTERNAL = 1
NUM_RANDOM = 1
SEED = 42 # Use a seed for reproducibility

print("Initializing environments...")

# Helper function to add a body segment safely
def add_body_segment(positions_deque, width, height):
    if not positions_deque:
        print("Warning: Cannot add body segment to empty positions deque.")
        return
    head = positions_deque[0]
    head_x, head_y = head

    # Try adding behind the head (relative to potential initial direction)
    # Example: Try adding one step left first
    potential_body_pos = (head_x - 1, head_y)
    if 0 <= potential_body_pos[0] < width and potential_body_pos != head:
         positions_deque.append(potential_body_pos)
         return

    # Try one step right
    potential_body_pos = (head_x + 1, head_y)
    if 0 <= potential_body_pos[0] < width and potential_body_pos != head:
         positions_deque.append(potential_body_pos)
         return

    # Try one step down
    potential_body_pos = (head_x, head_y - 1)
    if 0 <= potential_body_pos[1] < height and potential_body_pos != head:
         positions_deque.append(potential_body_pos)
         return

    # Try one step up
    potential_body_pos = (head_x, head_y + 1)
    if 0 <= potential_body_pos[1] < height and potential_body_pos != head:
         positions_deque.append(potential_body_pos)
         return

    print(f"Warning: Could not find a valid adjacent cell for head at {head} to add body segment.")


# Initialize FastSnakeEnv
try:
    fast_env = FastSnakeEnv(
        width=WIDTH,
        height=HEIGHT,
        num_apples=NUM_APPLES,
        max_rounds=MAX_ROUNDS,
        num_external_snakes=NUM_EXTERNAL,
        num_random_snakes=NUM_RANDOM
    )
    print("FastSnakeEnv initialized.")
    fast_env.reset(seed=SEED)
    print("FastSnakeEnv reset with seed.")

    # --- Add body segments manually to FastSnakeEnv ---
    print("Manually adding body segments to FastSnakeEnv snakes...")
    if fast_env.game and fast_env.external_snake_ids:
        ext_id = fast_env.external_snake_ids[0]
        if ext_id in fast_env.game.snakes:
            print(f"Adding body to external snake: {ext_id}")
            add_body_segment(fast_env.game.snakes[ext_id]['positions'], WIDTH, HEIGHT)
        else:
             print(f"Warning: External snake {ext_id} not found in fast_env.game.snakes")

    if fast_env.game and fast_env.random_snake_ids:
        rand_id = fast_env.random_snake_ids[0]
        if rand_id in fast_env.game.snakes:
            print(f"Adding body to random snake: {rand_id}")
            add_body_segment(fast_env.game.snakes[rand_id]['positions'], WIDTH, HEIGHT)
        else:
             print(f"Warning: Random snake {rand_id} not found in fast_env.game.snakes")
    else:
        print("Warning: No game object or random snakes found in fast_env.")

    # Update the internal board representation used by render_text
    if fast_env.game:
        fast_env.game._update_board()
        print("Called fast_env.game._update_board()")
    # --------------------------------------------------

    fast_state_text = fast_env.game_state_text()
    print("Generated game_state_text for FastSnakeEnv.")
except Exception as e:
    print(f"Error initializing or using FastSnakeEnv: {e}")
    import traceback
    traceback.print_exc() # Print traceback for debugging
    fast_state_text = None

# Initialize SnakeGameEnv
try:
    snake_env = SnakeGameEnv(
        width=WIDTH,
        height=HEIGHT,
        num_apples=NUM_APPLES,
        max_rounds=MAX_ROUNDS,
        num_external_snakes=NUM_EXTERNAL,
        num_random_snakes=NUM_RANDOM
    )
    print("SnakeGameEnv initialized.")
    snake_env.reset(seed=SEED)
    print("SnakeGameEnv reset with seed.")

    # --- Add body segments manually to SnakeGameEnv ---
    print("Manually adding body segments to SnakeGameEnv snakes...")
    if snake_env.game:
        # External snake ID is known ('external_1')
        ext_id_gym = snake_env.external_snake_ids[0] # Should be 'external_1'
        if ext_id_gym in snake_env.game.snakes:
            print(f"Adding body to external snake: {ext_id_gym}")
            add_body_segment(snake_env.game.snakes[ext_id_gym].positions, WIDTH, HEIGHT)
        else:
            print(f"Warning: External snake {ext_id_gym} not found in snake_env.game.snakes")

        # Random snake ID is known ('random_1')
        rand_id_gym = f"random_{1}"
        if rand_id_gym in snake_env.game.snakes:
             print(f"Adding body to random snake: {rand_id_gym}")
             add_body_segment(snake_env.game.snakes[rand_id_gym].positions, WIDTH, HEIGHT)
        else:
            print(f"Warning: Random snake {rand_id_gym} not found in snake_env.game.snakes")
    else:
        print("Warning: No game object found in snake_env.")
    # ---------------------------------------------------

    snake_state_text = snake_env.game_state_text()
    print("Generated game_state_text for SnakeGameEnv.")
except Exception as e:
    print(f"Error initializing or using SnakeGameEnv: {e}")
    import traceback
    traceback.print_exc() # Print traceback for debugging
    snake_state_text = None

print("\n" + "="*30)
print(" FastSnakeEnv Output ")
print("="*30)
if fast_state_text:
    print(fast_state_text)
    print(f"Length: {len(fast_state_text)}")
else:
    print("Failed to generate text.")

print("\n" + "="*30)
print(" SnakeGameEnv Output (Gold Standard)")
print("="*30)
if snake_state_text:
    print(snake_state_text)
    print(f"Length: {len(snake_state_text)}")
else:
    print("Failed to generate text.")

print("\n" + "="*30)
print(" Comparison ")
print("="*30)
if fast_state_text and snake_state_text:
    if fast_state_text == snake_state_text:
        print("Outputs are IDENTICAL.")
    else:
        print("Outputs differ.")
        # You could add more detailed diffing here if needed
        import difflib
        diff = difflib.unified_diff(
             fast_state_text.splitlines(keepends=True),
             snake_state_text.splitlines(keepends=True),
             fromfile='FastSnakeEnv',
             tofile='SnakeGameEnv',
         )
        print(''.join(diff))
elif not fast_state_text:
    print("Could not compare because FastSnakeEnv output failed.")
elif not snake_state_text:
    print("Could not compare because SnakeGameEnv output failed.")
