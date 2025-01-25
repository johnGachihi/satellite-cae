import os
import glob
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

def create_dataset_splits(root_dir, train_size=0.6, val_size=0.2, test_size=0.2):
    """
    Create train, validation, and test splits for a dataset organized in class subfolders.
    
    Parameters:
    root_dir (str): Root directory containing class subfolders
    train_size (float): Proportion of data for training
    val_size (float): Proportion of data for validation
    test_size (float): Proportion of data for testing
    """
    # Dictionary to store files for each class
    class_files = {}
    
    # Get all class folders
    class_folders = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
    
    # Create class to index mapping
    class_to_idx = {cls_name: idx for idx, cls_name in enumerate(sorted(class_folders))}
    
    # Collect files for each class
    for cls in class_folders:
        class_path = os.path.join(root_dir, cls)
        tif_files = glob.glob(os.path.join(class_path, "*.tif"))
        class_files[cls] = tif_files
    
    # Lists to store final splits
    train_data = []
    val_data = []
    test_data = []
    
    # Split each class maintaining proportions
    for cls, files in class_files.items():
        # First split into train and temp
        train_files, temp_files = train_test_split(
            files, 
            train_size=train_size,
            random_state=42,
            shuffle=True
        )
        
        # Split temp into val and test
        relative_val_size = val_size / (val_size + test_size)
        val_files, test_files = train_test_split(
            temp_files,
            train_size=relative_val_size,
            random_state=42,
            shuffle=True
        )
        
        # Add class index to each file
        cls_idx = class_to_idx[cls]
        train_data.extend([(f, cls_idx) for f in train_files])
        val_data.extend([(f, cls_idx) for f in val_files])
        test_data.extend([(f, cls_idx) for f in test_files])
    
    # Write to files
    def write_split(filename, data):
        with open(filename, 'w') as f:
            for file_path, cls_idx in data:
                abs_path = os.path.abspath(file_path)
                f.write(f"{abs_path} {cls_idx}\n")
    
    # Save splits to files
    write_split("train.txt", train_data)
    write_split("val.txt", val_data)
    write_split("test.txt", test_data)
    
    # Print statistics
    print(f"Total number of samples: {len(train_data) + len(val_data) + len(test_data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")
    print(f"Test samples: {len(test_data)}")
    
    # Print class distribution
    for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        print(f"\n{split_name} class distribution:")
        class_counts = {}
        for _, cls_idx in split_data:
            class_counts[cls_idx] = class_counts.get(cls_idx, 0) + 1
        for cls_idx, count in sorted(class_counts.items()):
            cls_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(cls_idx)]
            print(f"Class {cls_name} (index {cls_idx}): {count} samples")

if __name__ == "__main__":
    # Replace 'x' with your actual root directory path
    root_directory = "/home/ubuntu/satellite-cae/data/eurosat"
    create_dataset_splits(root_directory)
