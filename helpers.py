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
import pickle
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
    try:
        for col in data.columns:
            print(SEP)
            print(f'[{col}] {data[col][row]}')

    except KeyError:
        print(f'Row {row} not found in the dataset.')


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


def pickle_list_reviews(reviews_list: list, filepath: str) -> None:
    """Save a list of reviews to a pickle file.

    Args:
        reviews_list (list): The list of reviews to save.
        filepath (str): The path to the pickle file.

    Exceptions:
        FileNotFoundError: If the file path is not found.
    """

    try:
        with open(filepath, 'wb') as f:
            pickle.dump(reviews_list, f)
            print(f'Reviews pickled to {filepath}.')

    except FileNotFoundError as e:
        print(f'Error pickling reviews: {e}')


def load_pickled_reviews(filepath: str) -> list | None:
    """Load a list of reviews from a pickle file.

    Args:
        filepath (str): The path to the pickle file.

    Returns:
        list | None: The list of reviews.

    Exceptions:
        FileNotFoundError: If the file path is not found.
    """

    try:
        with open(filepath, 'rb') as f:
            reviews = pickle.load(f)
            print(f'Reviews loaded from {filepath}.')
            return reviews

    except FileNotFoundError as e:
        print(f'Error loading reviews: {e}')
        return None
