"""
Script: main.py
Description:
    This is the main script for the machine learning project.
"""

from argparse import ArgumentParser
import os

from loguru import logger

from config.variables import INPUT_DATA_DIR, OUTPUT_DATA_DIR
from src.data import load_data, save_data
from src.models import LinearRegression


def parse_arguments():
    """
    Parse command-line arguments for the script.
    """
    parser = ArgumentParser(description="Run the machine learning model.")
    parser.add_argument("--input", "-i", type=str, default="input_data.csv", help="Input data")
    parser.add_argument("--output", "-o", type=str, default="predictions.csv", help="Output data")
    parser.add_argument("--weight", "-w", type=float, default=1.0, help="Weight LR model")
    parser.add_argument("--intercept", "-b", type=float, default=0.0, help="Intercept  LR model")
    return parser.parse_args()


def main():
    """
    Main function to execute the machine learning model and print predictions.
    """
    logger.info("Starting ML Model inference")

    # Parse command-line arguments
    args = parse_arguments()
    input_data_file = os.path.join(INPUT_DATA_DIR, args.input)
    output_data_file = os.path.join(OUTPUT_DATA_DIR, args.output)
    weight = args.weight
    intercept = args.intercept

    # Initialize the model with command-line arguments
    logger.info("Init model")
    logger.debug(f"Params: {weight} and intercept: {intercept}")
    model = LinearRegression(weight=weight, intercept=intercept)

    # Load data
    logger.info("Loading data")
    data = load_data(input_data_file)

    # Make predictions
    logger.info("Making predictions")
    predictions = model.predict(data)
    logger.debug(f"Predictions: {predictions[:5]}...")

    # Save predictions
    logger.info("Saving predictions")
    save_data(output_data_file, predictions)

    # End of the script
    logger.info("Finished ML Model inference")

if __name__ == "__main__":
    main()
