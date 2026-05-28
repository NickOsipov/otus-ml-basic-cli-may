"""
Script: main.py
Description:
    This is the main script for the machine learning project.
"""

from argparse import ArgumentParser
import os

from config.variables import INPUT_DATA_DIR, OUTPUT_DATA_DIR
from src.data import load_data, save_data
from src.models import LinearRegression


def parse_arguments():
    """
    Parse command-line arguments for the script.
    """
    parser = ArgumentParser(description="Run the machine learning model.")
    parser.add_argument("--input", type=str, default="input_data.csv", help="Path to input data")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Path to save predictions")
    return parser.parse_args()


def main():
    """
    Main function to execute the machine learning model and print predictions.
    """
    # Parse command-line arguments
    args = parse_arguments()
    input_data_file = os.path.join(INPUT_DATA_DIR, args.input)
    output_data_file = os.path.join(OUTPUT_DATA_DIR, args.output)

    # Initialize the model (using dummy parameters for demonstration)
    model = LinearRegression(weight=1, intercept=0)

    # Load data
    data = load_data(input_data_file)

    # Make predictions
    predictions = model.predict(data)
    print(f"Predictions: {predictions}")

    # Save predictions
    save_data(output_data_file, predictions)

if __name__ == "__main__":
    main()
