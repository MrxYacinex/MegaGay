from clash_env import ClashRoyaleEnv
import cv2
import time
import numpy as np

def verify_vision():
    print("[INFO] Initializing Environment...")
    env = ClashRoyaleEnv()
    
    if env.model is None:
        print("[ERROR] Model not loaded! Check if best.pt exists.")
        return

    print("[INFO] Starting LIVE Vision Debugger.")
    print("[INFO] A window will open showing what the AI sees. Click the window and press 'q' to quit.")
    
    try:
        while True:
            cycle_start = time.time()
            
            # 1. Capture
            obs = env.controller.get_screenshot()
            if obs is None:
                continue
            
            # 2. Inference
            results = env.model(obs)
            
            # 3. Visualization
            # plot() returns a numpy array (BGR)
            annotated_frame = results[0].plot()
            
            # Add FPS counter
            fps = 1.0 / (time.time() - cycle_start + 1e-6)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 4. Show Window
            cv2.imshow("Clash Royale AI Vision", annotated_frame)
            
            # Exit on 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    verify_vision()
