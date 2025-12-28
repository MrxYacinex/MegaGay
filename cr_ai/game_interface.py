import logging
import time
import numpy as np
import cv2
from ppadb.client import Client as AdbClient

class GameController:
    def __init__(self, host="127.0.0.1", port=5037, device_serial=None):
        self.client = AdbClient(host=host, port=port)
        self.device = None
        self.device_serial = device_serial
        self.logger = logging.getLogger("GameController")
        logging.basicConfig(level=logging.INFO)

    def connect(self, bluestacks_port=5555):
        """Connects to the ADB server and specific device."""
        try:
            # Try to connect to BlueStacks specifically if not already connected
            result = self.client.remote_connect("127.0.0.1", bluestacks_port)
            if result:
                self.logger.info(f"Connected to 127.0.0.1:{bluestacks_port}")
        except Exception as e:
            self.logger.warning(f"Remote connect failed: {e}. Assuming already connected.")

        devices = self.client.devices()
        if not devices:
            self.logger.error("No devices found. Is BlueStacks running and ADB enabled?")
            return False

        if self.device_serial:
            self.device = next((d for d in devices if d.serial == self.device_serial), None)
        else:
            self.device = devices[0]

        if self.device:
            self.logger.info(f"Attached to device: {self.device.serial}")
            return True
        else:
            self.logger.error("Device not found.")
            return False

    def get_screenshot(self):
        """Captures a screenshot and returns it as a numpy array (OpenCV format)."""
        if not self.device:
            self.logger.error("Device not connected.")
            return None

        try:
            image_bytes = self.device.screencap()
            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            self.logger.error(f"Screenshot failed: {e}")
            return None

    def tap(self, x, y):
        """Simulates a tap at coordinates (x, y)."""
        if not self.device:
            return
        self.device.shell(f"input tap {x} {y}")

    def swipe(self, x1, y1, x2, y2, duration=300):
        """Simulates a swipe from (x1, y1) to (x2, y2)."""
        if not self.device:
            return
        self.device.shell(f"input swipe {x1} {y1} {x2} {y2} {duration}")

if __name__ == "__main__":
    # Test block
    controller = GameController()
    if controller.connect(bluestacks_port=5555): # Default BS port
        print("Connected!")
        img = controller.get_screenshot()
        if img is not None:
            print(f"Screenshot captured: {img.shape}")
            cv2.imwrite("test_screenshot.png", img)
            print("Saved to test_screenshot.png")
        else:
            print("Failed to capture screenshot.")
    else:
        print("Failed to connect.")
