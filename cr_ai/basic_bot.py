import time
import random
from clash_env import ClashRoyaleEnv

def run_bot():
    print("Starting Battle Bot...")
    try:
        env = ClashRoyaleEnv()
    except Exception as e:
        print(f"Failed to start env: {e}")
        return

    obs, _ = env.reset()
    print(f"Bot initialized. Screen size: {env.screen_width}x{env.screen_height}")
    
    steps = 0
    try:
        while True:
            steps += 1
            print(f"\n--- Step {steps} ---")
            
            # --- Logic: The "Brain" ---
            # For now: Randomly pick a card (0-3) and place it on the enemy side
            # Action format: [card_idx, x_pct, y_pct]
            
            card_idx = random.randint(0, 3)
            
            # Smart Placement (Simple Rule):
            # Place mostly on the bridge for pressure
            # X: Left Bridge ~25%, Right Bridge ~75%
            # Y: River is roughly ~50%. Place slightly above (45%) or below (55%).
            
            lane = random.choice(["left", "right"])
            if lane == "left":
                x_pos = random.randint(20, 30) # Left lane
            else:
                x_pos = random.randint(70, 80) # Right lane
            
            y_pos = random.randint(50, 65) # Middle/Enemy side
            
            print(f"Playing Card {card_idx+1} on {lane} lane ({x_pos}%, {y_pos}%)")
            
            action = [card_idx, x_pos, y_pos]
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Wait for Elixir (Crude logic: just wait 3-4 seconds before next move)
            # A real bot would read the Elixir bar.
            wait_time = random.uniform(3.0, 5.0)
            print(f"Waiting {wait_time:.1f}s for elixir...")
            time.sleep(wait_time)

    except KeyboardInterrupt:
        print("Bot stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    run_bot()
