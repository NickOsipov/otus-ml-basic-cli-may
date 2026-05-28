"""
Script: main.py
Description:
    This is the main script for the machine learning project.
"""

import os

from src.data import load_data, save_data
from src.models import LinearRegression


DATA_DIR = "data"
INPUT_DATA = os.path.join(DATA_DIR, "input")
OUTPUT_DATA = os.path.join(DATA_DIR, "output")

def main():
    """
    Main function to execute the machine learning model and print predictions.
    """
    model = LinearRegression(weight=1, intercept=0)

    input_data_file = os.path.join(INPUT_DATA, "data.csv")
    data = load_data(input_data_file)

    predictions = model.predict(data)
    print(f"Predictions: {predictions}")

    output_data_file = os.path.join(OUTPUT_DATA, "predictions.csv")
    save_data(output_data_file, predictions)

if __name__ == "__main__":
    main()
