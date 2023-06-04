import pandas as pd
import random
import math
import argparse
import os

def split_dataset(filename, train_ratio):
    df = pd.read_csv(filename)

    # Shuffle the dataset
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    train_size = math.floor(len(df) * train_ratio)
    remaining_data = df.iloc[train_size:]

    val_size = test_size = math.floor(len(remaining_data) / 2)

    train_data = df.iloc[:train_size]
    val_data = remaining_data.iloc[:val_size]
    test_data = remaining_data.iloc[val_size:]

    base_filename = os.path.splitext(os.path.basename(filename))[0]
    output_dir = os.path.dirname(filename)

    train_filename = os.path.join(output_dir, base_filename + '_train.csv')
    val_filename = os.path.join(output_dir, base_filename + '_val.csv')
    test_filename = os.path.join(output_dir, base_filename + '_test.csv')

    train_data.to_csv(train_filename, index=False)
    val_data.to_csv(val_filename, index=False)
    test_data.to_csv(test_filename, index=False)

    print("Dataset split completed.")
    print("Train data saved as:", train_filename)
    print("Validation data saved as:", val_filename)
    print("Test data saved as:", test_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Split a CSV dataset into train, validation, and test sets.')
    parser.add_argument('filename', type=str, help='the filename of the CSV dataset')
    parser.add_argument('train_ratio', type=float, help='the split ratio for the train set')
    
    args = parser.parse_args()

    split_dataset(args.filename, args.train_ratio)
