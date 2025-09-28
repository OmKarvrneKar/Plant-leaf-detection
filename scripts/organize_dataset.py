import os
import shutil
from pathlib import Path
import random
from sklearn.model_selection import train_test_split

def organize_plantvillage_dataset():
    """Organize PlantVillage dataset for training"""
    
    # Common possible source directories (check which one you have)
    possible_sources = [
        Path("plantvillage dataset/color"),  # Added for your specific folder structure
        Path("PlantVillage"),
        Path("plantvillage-dataset/PlantVillage"), 
        Path("color"),
        Path("plantvillage-dataset/color"),
        Path("New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"),
        Path(".")  # If disease folders are directly in current directory
    ]
    
    source_dir = None
    
    # Find the correct source directory
    for possible_dir in possible_sources:
        if possible_dir.exists():
            # Check if it contains disease folders
            disease_folders = [f for f in possible_dir.iterdir() if f.is_dir() and ('___' in f.name or 'healthy' in f.name.lower())]
            if disease_folders:
                source_dir = possible_dir
                print(f"✅ Found dataset in: {source_dir}")
                break
    
    if not source_dir:
        print("❌ Could not find PlantVillage dataset folders!")
        print("📁 Current directory contents:")
        for item in Path(".").iterdir():
            print(f"   {item}")
        return False
    
    # Output directories
    output_base = Path("plantvillage_organized")
    train_dir = output_base / "train"
    val_dir = output_base / "val"
    
    # Remove existing organized dataset
    if output_base.exists():
        shutil.rmtree(output_base)
    
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    
    print(f"🔄 Organizing dataset from: {source_dir}")
    print(f"📁 Output directory: {output_base}")
    
    total_classes = 0
    total_images = 0
    
    # Process each disease class folder
    for class_folder in source_dir.iterdir():
        if class_folder.is_dir():
            class_name = class_folder.name
            
            # Skip non-disease folders
            if not ('___' in class_name or 'healthy' in class_name.lower()):
                continue
                
            print(f"📊 Processing: {class_name}")
            
            # Find all images in this class
            image_files = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
                image_files.extend(list(class_folder.glob(ext)))
            
            if not image_files:
                print(f"   ⚠️ No images found in {class_name}")
                continue
            
            # Create class directories
            train_class_dir = train_dir / class_name
            val_class_dir = val_dir / class_name
            train_class_dir.mkdir(exist_ok=True)
            val_class_dir.mkdir(exist_ok=True)
            
            # Split images 80% train, 20% validation
            train_files, val_files = train_test_split(
                image_files, 
                test_size=0.2, 
                random_state=42
            )
            
            # Copy training images
            for i, img_file in enumerate(train_files):
                dst_file = train_class_dir / f"train_{i:04d}{img_file.suffix}"
                shutil.copy2(img_file, dst_file)
            
            # Copy validation images
            for i, img_file in enumerate(val_files):
                dst_file = val_class_dir / f"val_{i:04d}{img_file.suffix}"
                shutil.copy2(img_file, dst_file)
            
            total_classes += 1
            total_images += len(image_files)
            print(f"   ✅ {len(train_files)} train, {len(val_files)} val images")
    
    print(f"\n🎉 Dataset organization complete!")
    print(f"📊 Total classes: {total_classes}")
    print(f"📊 Total images: {total_images:,}")
    print(f"📁 Organized dataset: {output_base}")
    
    return True

if __name__ == "__main__":
    print("🌾 PLANTVILLAGE DATASET ORGANIZER")
    print("=" * 50)
    
    success = organize_plantvillage_dataset()
    
    if success:
        print("\n🚀 READY FOR TRAINING!")
        print("📋 Next command:")
        print("python finetune_model.py --data_dir plantvillage_organized --epochs 8 --batch_size 16 --output_dir plantvillage_model")
    else:
        print("\n❌ Please check your dataset structure and try again")