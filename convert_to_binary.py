import os, shutil, random
from pathlib import Path

random.seed(42)

# Run this script from: E:\chaintogether\SIH\plant_leaf_project
SOURCE = Path("New_Plant_Diseases_Dataset")
TARGET = Path("dataset")

# Create target directories
for split in ["train", "val"]:
    for label in ["healthy", "diseased"]:
        (TARGET / split / label).mkdir(parents=True, exist_ok=True)

def is_healthy(name: str) -> bool:
    return "healthy" in name.lower()
  
def copy_images(source_dir: Path, target_split: str):
    """Copy images from source_dir to target_split (train or val)"""
    total = 0
    for class_dir in source_dir.iterdir():
        if not class_dir.is_dir():
            continue
            
        # Get all image files
        images = [f for f in class_dir.glob("*") 
                 if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        
        if not images:
            continue
            
        # Determine label
        label = "healthy" if is_healthy(class_dir.name) else "diseased"
        target_dir = TARGET / target_split / label
        
        # Copy images
        for img in images:
            new_name = f"{class_dir.name}_{img.name}"
            shutil.copy2(img, target_dir / new_name)
            total += 1
    
    return total

# Copy from train and val directories
train_count = copy_images(SOURCE / "train", "train")
val_count = copy_images(SOURCE / "val", "val")

print(f"✅ Copied {train_count} images to dataset/train")
print(f"✅ Copied {val_count} images to dataset/val")
print(f"✅ Total: {train_count + val_count} images processed")