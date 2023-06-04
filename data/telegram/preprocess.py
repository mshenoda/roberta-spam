import pandas as pd
import argparse

def remove_newlines(input_file, output_file):
    df = pd.read_csv(input_file)
    df = df.replace('\n', '', regex=True)
    df.to_csv(output_file, index=False)
    print("Newlines removed successfully!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remove newlines from each row in a CSV file')
    parser.add_argument('input', help='Input CSV file')
    parser.add_argument('output', help='Output CSV file')
    args = parser.parse_args()

    remove_newlines(args.input, args.output)
