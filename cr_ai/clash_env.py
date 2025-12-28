import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2
import time
from pathlib import Path

from game_interface import GameController
from knowledge_base import KnowledgeBase
# from ultralytics import YOLO  # Uncomment when model is ready

class ClashRoyaleEnv(gym.Env):
    """
    Custom Environment that follows gym interface.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, device_serial=None, host="127.0.0.1", bluestacks_port=5555):
        super(ClashRoyaleEnv, self).__init__()
        
        # 1. Initialize Game Controller (ADB)
        self.controller = GameController(host=host, device_serial=device_serial)
        if not self.controller.connect(bluestacks_port=bluestacks_port):
            raise ConnectionError("Could not connect to BlueStacks/Device.")
            
        # 2. Initialize Knowledge Base
        self.kb = KnowledgeBase()
        
        # 3. Initialize Vision Model (Placeholder)
        # self.model = YOLO("cr_ai/runs/detect/train/weights/best.pt") # Example path
        self.model = None

        # Define Action Space:
        # Action: [Card_Index (0-3), X_Position (0-100), Y_Position (0-100)]
        # We discretize the board into a grid (e.g., 10x10 or finer)
        # For simplicity v1: 4 cards * 2 lanes = 8 discrete actions? 
        # Better: MultiDiscrete([4, 18, 32]) -> Card 0-3, Grid X (18 tiles), Grid Y (32 tiles)
        self.action_space = spaces.MultiDiscrete([4, 100, 100])

        # Define Observation Space:
        # Image (Screen) + Scalar Features (Elixir, Time, etc.)
        # For v1, we just return the image.
        self.observation_space = spaces.Box(low=0, high=255, shape=(540, 960, 3), dtype=np.uint8)

    def reset(self, seed=None, options=None):       
        super().reset(seed=seed)
        # Logic to restart the match (e.g., click "Battle" button)
        # For now, we assume the user is manually starting/restarting or in training camp loop
        
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        """
        Action format: [card_index, x_percent, y_percent]
        """
        card_idx, x_pct, y_pct = action
        
        # 1. Select Card
        # Map card_idx 0-3 to screen coordinates (Bottom UI)
        self._tap_card(card_idx)
        time.sleep(0.1)
        
        # 2. Place Card
        # Map percent to pixel coordinates
        screen_x = int((x_pct / 100.0) * 960)
        screen_y = int((y_pct / 100.0) * 540)
        self.controller.tap(screen_x, screen_y)
        
        # 3. Wait for game update
        time.sleep(2.0) # Game allows one move per ~2s? Setup delay
        
        # 4. Get new state
        observation = self._get_observation()
        
        # 5. Calculate Reward (Placeholder)
        # Need to detect tower health delta
        reward = 0 
        terminated = False
        truncated = False
        info = {}
        
        return observation, reward, terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass

    def _get_observation(self):
        img = self.controller.get_screenshot()
        if img is None:
            # Return black screen if failed
            return np.zeros((540, 960, 3), dtype=np.uint8)
        return img

    def _tap_card(self, card_idx):
        # Hardcoded coordinates for 960x540 resolution
        # Cards are roughly at Y=480, distributed across X
        card_x_positions = [300, 420, 540, 660] # Approx centers
        if 0 <= card_idx < 4:
            self.controller.tap(card_x_positions[card_idx], 500)

if __name__ == "__main__":
    # Test the environment
    try:
        env = ClashRoyaleEnv()
        obs, info = env.reset()
        print(f"Env Reset! Obs shape: {obs.shape}")
        
        # Random action
        action = env.action_space.sample()
        print(f"Taking random action: {action}")
        obs, reward, term, trunc, info = env.step(action)
        print("Step done!")
    except Exception as e:
        print(f"Env failed: {e}")
