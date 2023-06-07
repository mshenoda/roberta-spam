import argparse
import pandas as pd
import matplotlib.pyplot as plt

def visualize_class_distribution(csv_file, class_column, title):
    # Load the CSV dataset using pandas
    df = pd.read_csv(csv_file)

    # Count the occurrences of each class
    class_counts = df[class_column].value_counts()

    # Plot the class distribution
    plt.figure(figsize=(10, 6))
    class_counts.plot(kind='bar')
    plt.title(title) #'Class Distribution'
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Visualize class distribution in a CSV file.')
    parser.add_argument('csv_file', type=str, help='Path to the CSV file')
    parser.add_argument('class_column', type=str, help='Name of the column containing class labels')
    parser.add_argument('title', type=str, help='Plot Title')

    # Parse command-line arguments
    args = parser.parse_args()

    # Call the visualization function
    visualize_class_distribution(args.csv_file, args.class_column, args.title)
