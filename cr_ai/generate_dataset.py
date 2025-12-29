import sys
from pathlib import Path
import numpy as np

# Add KataCR to path
sys.path.append(str(Path("KataCR").resolve()))

from katacr.build_dataset.generator import Generator

def generate_dataset(num_images=50):
    # Initialize generator
    # intersect_ratio_thre: prevent overlapping
    gen = Generator(seed=42, intersect_ratio_thre=0.6, augment=True)
    
    output_dir = Path("dataset_yolo")
    images_dir = output_dir / "images"
    labels_dir = output_dir / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating {num_images} images to {output_dir}...")
    
    for i in range(num_images):
        gen.reset()
        gen.add_tower()
        gen.add_unit(n=random.randint(10, 30)) # Random number of units
        
        # Build image and boxes
        # box format from generator is cx, cy, w, h (normalized) if box_format='cxcywh'
        # box array: [cx, cy, w, h, state..., cls]
        # We need to verify if cls is the last column. 
        # In generator.py: box.append((*u.xyxy, *u.states, u.cls)) -> then converted to cxcywh
        
        # shape: (origin_img, box, pil_img)
        _, boxes, img = gen.build(verbose=False, show_box=False, box_format='cxcywh')
        
        # Save image
        img_name = f"gen_{i:05d}.jpg"
        img.save(images_dir / img_name)
        
        # Save label
        txt_name = f"gen_{i:05d}.txt"
        with open(labels_dir / txt_name, "w") as f:
            for b in boxes:
                # b is [cx, cy, w, h, ... , cls]
                # We assume cls is the LAST element [-1].
                cls_id = int(b[-1])
                cx, cy, w, h = b[:4]
                
                # YOLO format: class cx cy w h
                f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
        
        if (i+1) % 10 == 0:
            print(f"Generated {i+1}/{num_images}")

    print("Done!")

import random
if __name__ == "__main__":
if __name__ == "__main__":
    generate_dataset(1000) # Generate 1000 samples for better training
