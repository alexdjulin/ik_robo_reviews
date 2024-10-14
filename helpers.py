#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File Name: helpers.py
Author: Alexandre Donciu-Julin
Date: 2024-10-14
Description: Helper methods to interact with the dataset quickly.
"""

# Import statements
import os
import re
import pandas as pd
from random import randint
from tqdm import tqdm
tqdm.pandas()


SEP = 100 * '-'


def print_random_product_sheet(data: pd.DataFrame) -> None:
    """Print a random product sheet from the dataset.

    Args:
        data (pd.DataFrame): The dataset to print a random product sheet from.
    """
    row = randint(0, data.shape[0])
    for col in data.columns:
        print(SEP)
        print(f'[{col}] {data[col][row]}')


def load_pickled_dataset(file_path: str) -> pd.DataFrame | None:
    """Load the dataset from a pickle file.

    Args:
        file_path (str): The path to the pickled dataset file.

    Returns:
        pd.DataFrame | None: The dataset as a pandas DataFrame.

    Raises:
        FileNotFoundError: If the pickled dataset file is not found.
    """

    try:
        data = pd.read_pickle(file_path)
        print(f'Dataset loaded from {file_path}.')
        return data

    except FileNotFoundError:
        print(f'No pickled dataset found at {file_path}.')
        return None
