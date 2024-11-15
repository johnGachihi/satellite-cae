#!/usr/bin/env python3

import os
import csv
import shutil
import argparse
from pathlib import Path

SPLITS = {
    'train': 'flood_train_data.csv',
    'val': 'flood_valid_data.csv',
    'test': 'flood_test_data.csv'
}

def copy_files(source_dir, dest_dir, file_list):
    for file in file_list:
        source_file = source_dir / file
        if source_file.exists():
            shutil.copy2(source_file, dest_dir)
        else:
            print(f"Warning: File not found - {source_file}")

def convert_s1_to_s2(filename):
    parts = filename.split('_')
    if len(parts) >= 3 and parts[-1].startswith('S1Hand'):
        parts[-1] = 'S2Hand' + parts[-1][6:]
    return '_'.join(parts)

def process_split(root_path, split_name, csv_file):
    splits_path = root_path / 'v1.1/splits/flood_handlabeled'
    input_path = root_path / 'v1.1/data/flood_events/HandLabeled/S2Hand'
    label_path = root_path / 'v1.1/data/flood_events/HandLabeled/LabelHand'

    with open(splits_path / csv_file, 'r') as f:
        reader = csv.reader(f)
        input_files, label_files = zip(*reader)
    input_files = tuple(convert_s1_to_s2(filename) for filename in input_files)

    for data_type, files in [('s2', input_files), ('labelhand', label_files)]:
        new_dir = root_path / split_name / data_type
        new_dir.mkdir(parents=True, exist_ok=True)
        source_dir = input_path if data_type == 's2' else label_path
        copy_files(source_dir, new_dir, files)

def main(root_path):
    root_path = Path(root_path).resolve()
    print(f"Processing dataset in: {root_path}")

    for split_name, csv_file in SPLITS.items():
        process_split(root_path, split_name, csv_file)
    
    print("Dataset splitting completed successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split dataset into train, val, and test sets.")
    parser.add_argument("root_path", help="Root path of the dataset")
    args = parser.parse_args()

    main(args.root_path)
