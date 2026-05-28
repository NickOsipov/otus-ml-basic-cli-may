"""
Module: variables.py
Description:
    This module defines constants for the data directory and file paths
    used in the machine learning project.
"""

import os

from dotenv import load_dotenv

load_dotenv()


DATA_DIR = "data"
INPUT_DATA_DIR = os.path.join(DATA_DIR, "input")
OUTPUT_DATA_DIR = os.path.join(DATA_DIR, "output")

MODEL_TYPE = os.getenv("MODEL_TYPE")

