"""
Script: main.py
Description:
    This is the main script for the machine learning project.
"""

from src.models import LinearRegression


def main():
    """
    Main function to execute the machine learning model and print predictions.
    """
    model = LinearRegression(weight=1, intercept=0)
    data = [1, 2, 3]
    predictions = model.predict(data)
    print(f"Predictions: {predictions}")

if __name__ == "__main__":
    main()
