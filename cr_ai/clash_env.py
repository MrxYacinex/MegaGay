import gymnasium as gym
import numpy as np
from gymnasium import spaces
import cv2
import time
from pathlib import Path

from game_interface import GameController
from knowledge_base import KnowledgeBase
from ultralytics import YOLO  # Uncomment when model is ready

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
        
        # 3. Initialize Vision Model
        # Load the locally trained model
        model_path = Path(__file__).parent.parent / "runs/detect/train/weights/best.pt"
        if model_path.exists():
            print(f"[INFO] Loading YOLO model from {model_path}")
            self.model = YOLO(model_path)
        else:
            print(f"[WARNING] Model not found at {model_path}. Using lightweight fallback or None.")
            self.model = None

        # Capture one frame to determine resolution
        self.screen_width = 1080
        self.screen_height = 1920
        init_screen = self.controller.get_screenshot()
        if init_screen is not None:
            # cv2 image is (Height, Width, Channels)
            self.screen_height, self.screen_width = init_screen.shape[:2]
            print(f"[INFO] Detect resolution: {self.screen_width}x{self.screen_height}")

        # Define Action Space:
        # MultiDiscrete([4, 100, 100]) -> Card 0-3, Grid X (0-100%), Grid Y (0-100%)
        self.action_space = spaces.MultiDiscrete([4, 100, 100])

        # Define Observation Space:
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3), dtype=np.uint8)

    def reset(self, seed=None, options=None):       
        super().reset(seed=seed)
        observation = self._get_observation()
        info = {}
        return observation, info

    def step(self, action):
        """
        Action format: [card_index, x_percent, y_percent]
        """
        card_idx, x_pct, y_pct = action
        
        # 1. Select Card
        self._tap_card(card_idx)
        time.sleep(0.2)
        
        # 2. Place Card
        # Map percent to pixel coordinates
        screen_x = int((x_pct / 100.0) * self.screen_width)
        screen_y = int((y_pct / 100.0) * self.screen_height)
        
        # Safety clip
        screen_x = max(0, min(screen_x, self.screen_width - 1))
        screen_y = max(0, min(screen_y, self.screen_height - 1))
        
        self.controller.tap(screen_x, screen_y)
        
        # 3. Wait for game update (placement delay)
        time.sleep(1.0) 
        
        # 4. Get new state
        observation = self._get_observation()
        
        # 5. Calculate Reward (Placeholder)
        reward = 0 
        terminated = False
        truncated = False
        info = {}
        
        return observation, reward, terminated, truncated, info

    def _get_observation(self):
        img = self.controller.get_screenshot()
        if img is None:
            return np.zeros((self.screen_height, self.screen_width, 3), dtype=np.uint8)
        return img

    def _tap_card(self, card_idx):
        # Calculate dynamic positions based on resolution
        # Cards are roughly at 90% height
        card_y = int(self.screen_height * 0.90)
        
        # X positions for 4 cards (centered in their slots)
        # Slots are roughly: 17%, 39%, 61%, 83% width
        x_ratios = [0.17, 0.39, 0.61, 0.83]
        
        if 0 <= card_idx < 4:
            card_x = int(self.screen_width * x_ratios[card_idx])
            self.controller.tap(card_x, card_y)

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
