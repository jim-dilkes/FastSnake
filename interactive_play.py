# interactive_play.py

import sys
import tty
import termios
# Adjusted imports for running from modules/SnakeBench directory
from backend.env import FastSnakeEnv
from backend.core import UP, DOWN, LEFT, RIGHT # Import constants

# --- Game Parameters ---
WIDTH = 10
HEIGHT = 10
NUM_APPLES = 3
MAX_ROUNDS = 200 # Allow for longer interactive play
NUM_RANDOM_SNAKES = 1
# ---------------------

# --- Action Mapping ---
# Map keyboard input (lowercase) to game actions
input_to_action = {
    'w': UP,    # Use W for UP
    's': DOWN,  # Use S for DOWN
    'a': LEFT,  # Use A for LEFT
    'd': RIGHT, # Use D for RIGHT
    'q': 'quit' # Add a quit option
}
# --------------------

# Helper function to get single character input without needing Enter
def getch():
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(sys.stdin.fileno())
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    return ch

def play_game():
    print("Initializing FastSnakeEnv for interactive play...")
    env = FastSnakeEnv(
        width=WIDTH,
        height=HEIGHT,
        num_apples=NUM_APPLES,
        max_rounds=MAX_ROUNDS,
        num_external_snakes=1, # Must be 1 for this script
        num_random_snakes=NUM_RANDOM_SNAKES
    )

    print("Resetting environment...")
    obs, info = env.reset() # Get initial state
    your_snake_id = env.external_snake_ids[0]
    done = False
    total_reward = 0

    print("\nStarting game!")
    print("Controls: W=Up, S=Down, A=Left, D=Right, Q=Quit")

    while not done:
        # 1. Render state
        print("\n" + "="*40)
        try:
            print(env.game_state_text())
            current_score = info.get('scores', {}).get(your_snake_id, 0)
            print(f"Round: {info.get('round', 0)}")
            print(f"Your Score ({your_snake_id}): {current_score}")
            print(f"Total Reward: {total_reward:.2f}")
            print("Enter move (w/a/s/d) or q to quit: ", end='', flush=True)
        except Exception as e:
            print(f"Error rendering state: {e}")
            break # Exit if rendering fails

        # 2. Get user input
        action = None
        while action is None:
            try:
                char = getch().lower()
            except Exception as e:
                 print(f"\nError reading input: {e}. Exiting.")
                 return # Exit if input reading fails

            if char in input_to_action:
                move = input_to_action[char]
                if move == 'quit':
                    print("\nQuitting game.")
                    return
                else:
                    action = move
                    print(char) # Echo valid move
            else:
                print(f"\nInvalid input '{char}'. Use w/a/s/d or q.", end='', flush=True)
                # Reprint prompt after handling invalid input without newline
                print("\nEnter move (w/a/s/d) or q to quit: ", end='', flush=True)


        # 3. Step environment
        try:
            # Since num_external_snakes is 1, step expects a single action integer
            # obs, reward, done, _, info = env.step(action) # Standard Gym returns 5 values
            # FastSnakeEnv step returns 5 values: obs, reward, done, False, info
            obs, reward, game_finished, truncated, info = env.step(action)
            done = game_finished # Use the correct done flag

            total_reward += reward
            print(f"Action taken. Reward this step: {reward:.2f}")
        except Exception as e:
            print(f"\nError during environment step: {e}")
            import traceback
            traceback.print_exc()
            break # Exit if step fails

    # 4. Game over
    print("\n" + "="*40)
    print(" GAME OVER! ".center(40, "="))
    print("="*40)
    try:
        # Print final state
        print("Final State:")
        print(env.game_state_text())
        final_scores = info.get('all_scores', {})
        print("\nFinal Scores:")
        for snake_id, score in final_scores.items():
             print(f"  - {snake_id}: {score}")
        print(f"\nTotal reward for your snake ({your_snake_id}): {total_reward:.2f}")
        print(f"Game lasted {info.get('round', 'N/A')} rounds.")
    except Exception as e:
        print(f"Error printing final state: {e}")

if __name__ == "__main__":
    play_game()
