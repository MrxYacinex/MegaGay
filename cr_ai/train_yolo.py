from ultralytics import YOLO
import sys
from pathlib import Path

# Add KataCR to path to get class names
# We need to add the parent of 'katacr' package, which is 'cr_ai/KataCR'
script_dir = Path(__file__).parent
sys.path.append(str((script_dir / "KataCR").resolve()))
from katacr.constants.label_list import unit_list

def train_yolo_on_mac():
    # 1. Define Paths
    script_dir = Path(__file__).parent
    base_dir = (script_dir / "dataset_yolo").resolve()
    yaml_path = base_dir / "data.yaml"
    
    # Ensure directory exists
    if not base_dir.exists():
        print(f"[ERROR] Dataset directory not found at: {base_dir}")
        print("Did you run generate_dataset.py?")
        return
    
    # 2. Create data.yaml content
    # YOLO needs a definition file telling it where images are and what classes exist
    # We use the 'unit_list' from KataCR which matches the generator IDs
    
    classes_yaml = ""
    for idx, name in enumerate(unit_list):
        classes_yaml += f"  {idx}: {name}\n"
    
    yaml_content = f"""
path: {base_dir}
train: images
val: images

names:
{classes_yaml}
"""
    
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"[INFO] Created YOLO config at {yaml_path}")
    
    # 3. Initialize Model
    # 'yolov8n.pt' is the 'nano' version (smallest/fastest). Good for realtime on CPU/MPS.
    print("[INFO] Loading YOLOv8 Nano model...")
    model = YOLO("yolov8n.pt") 
    
    # 4. Train Model
    # Device: 'mps' is for Mac M1/M2 Metal Performance Shaders (GPU acceleration)
    # If mps fails, use 'cpu'
    device = 'mps' 
    print(f"[INFO] Starting training device={device}...")
    
    try:
        results = model.train(
            data=str(yaml_path),
            epochs=50,       # Start with 50 epochs
            imgsz=640,       # Image size (generator uses ~640x...)
            device=device,   
            batch=16,        
            plots=True
        )
        
        # 5. Export
        print("[INFO] Training finished. Exporting best model...")
        success = model.export(format="onnx")
        print(f"[SUCCESS] Exported to: {success}")
        
    except Exception as e:
        print(f"[ERROR] Training failed with device='{device}': {e}")
        print("[INFO] Fallback to CPU training...")
        model.train(data=str(yaml_path), epochs=5, imgsz=640, device='cpu')

if __name__ == "__main__":
    train_yolo_on_mac()
