import argparse
import pandas as pd

def merge_csv(input_files, output_file):
    dfs = []
    for file in input_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    merged_df = pd.concat(dfs)
    merged_df.to_csv(output_file, index=False)
    print("Merged CSV files saved successfully as", output_file)

if __name__ == "__main__":
    # python merge_csv.py input1.csv input2.csv input3.csv output.csv
    parser = argparse.ArgumentParser(description="Merge multiple CSV files into a single CSV output file.")
    parser.add_argument("input_files", nargs="+", help="Input CSV files to merge")
    parser.add_argument("output_file", help="Output CSV file")
    args = parser.parse_args()

    merge_csv(args.input_files, args.output_file)
