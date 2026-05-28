"""
Module: data.py
Description:
    This module contains functions for loading and saving data from/to CSV files.
"""


import csv
from typing import List, Union

def load_data(file_path: str) -> List[float]:
    """
    Load data from a CSV file and return it as a list of floats.

    Parameters
    ----------
    file_path : str
        The path to the CSV file containing the data.

    Returns
    -------
    list of float
        A list of data values loaded from the CSV file. If there is an error during conversion
    """
    with open(file_path, newline='') as csvfile:
        reader = csv.reader(csvfile)
        try:
            data = [float(row[0]) for row in reader]
            return data
        except ValueError as e:
            print(f"Error converting data to float: {e}")
            return []

def save_data(file_path: str, data: List[Union[float, int]]):
    """
    Save a list of data values to a CSV file.

    Parameters
    ----------
    file_path : str
        The path to the CSV file where the data will be saved.
    data : list of float or int
        A list of data values to be saved to the CSV file.  

    Returns
    -------
    None
    """
    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for value in data:
            writer.writerow([value])
