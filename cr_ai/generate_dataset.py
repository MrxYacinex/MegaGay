import sys
from pathlib import Path
import numpy as np
import random

# Add KataCR to path
# Use parent directory relative to this script
sys.path.append(str((Path(__file__).parent / "KataCR").resolve()))

from katacr.build_dataset.generator import Generator

def generate_dataset(num_images=500):
    # Initialize generator
    gen = Generator(seed=42, intersect_ratio_thre=0.6, augment=True)
    
    base_dir = Path("dataset_yolo")
    
    # Define directories
    dirs = {
        "train": {
            "images": base_dir / "images" / "train",
            "labels": base_dir / "labels" / "train"
        },
        "val": {
            "images": base_dir / "images" / "val",
            "labels": base_dir / "labels" / "val"
        }
    }
    
    # Create directories
    for split in dirs:
        dirs[split]["images"].mkdir(parents=True, exist_ok=True)
        dirs[split]["labels"].mkdir(parents=True, exist_ok=True)
        
    print(f"Generating {num_images} images to {base_dir} with 80/20 split...")
    
    for i in range(num_images):
        # Determine split
        split = "train" if i < num_images * 0.8 else "val"
        current_images_dir = dirs[split]["images"]
        current_labels_dir = dirs[split]["labels"]

        gen.reset()
        gen.add_tower()
        gen.add_unit(n=random.randint(5, 25)) # Random number of units
        
        # Build
        _, boxes, img = gen.build(verbose=False, show_box=False, box_format='cxcywh')
        
        # Save image
        img_name = f"gen_{i:05d}.jpg"
        img.save(current_images_dir / img_name)
        
        # Save label
        txt_name = f"gen_{i:05d}.txt"
        with open(current_labels_dir / txt_name, "w") as f:
            for b in boxes:
                cls_id = int(b[-1])
                cx, cy, w, h = b[:4]
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        
        if (i+1) % 50 == 0:
            print(f"Generated {i+1}/{num_images}")

    print("Done!")

if __name__ == "__main__":
    generate_dataset(500) # Generate 500 samples for better training
