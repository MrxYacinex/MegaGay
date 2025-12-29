from clash_env import ClashRoyaleEnv
import cv2
import time
import numpy as np

def verify_vision():
    print("[INFO] Initializing Environment...")
    # This will connect to ADB and load the YOLO model
    env = ClashRoyaleEnv()
    
    if env.model is None:
        print("[ERROR] Model not loaded! Check if best.pt exists.")
        return

    print("[INFO] capturing screenshot...")
    # Reset gets the first observation
    obs, _ = env.reset()
    
    # Run Inference
    print("[INFO] Running YOLO inference...")
    results = env.model(obs)
    
    # Results is a list (one per image)
    for r in results:
        # Plot results on the image
        # params: conf=True (show confidence), labels=True (show labels)
        im_array = r.plot()  
        
        # Save validation image
        output_path = "vision_test_result.jpg"
        cv2.imwrite(output_path, np.array(im_array))
        print(f"[SUCCESS] Saved detection result to {output_path}")
        print(f"[INFO] Detected {len(r.boxes)} objects.")
        
        # Print what was found
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = r.names[cls_id]
            print(f" - Found: {name} ({conf:.2f})")

if __name__ == "__main__":
    verify_vision()
