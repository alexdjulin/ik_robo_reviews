#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
File Name: helpers.py
Author: Alexandre Donciu-Julin
Date: 2024-10-14
Description: Helper methods to interact with the dataset.
"""

# Import statements --------------------------------------------------------------------------------
import re
import pickle
import pandas as pd
import random
from functools import reduce
from tqdm import tqdm
tqdm.pandas()


# Constants and paths ------------------------------------------------------------------------------
DATA_PROCESSED_PKL = 'pickle/data_processed.pkl'
DATA_CLUSTERED_PKL = 'pickle/data_clustered.pkl'
DATA_SA_PKL = 'pickle/data_sentiment_analysis.pkl'
DATA_SCORED_PKL = 'pickle/data_scored.pkl'
SEP = 100 * '-'


# Helper methods -----------------------------------------------------------------------------------

def load_pickled_dataset(filepath: str) -> pd.DataFrame | None:
    """Load a pickled dataset from a file.

    Args:
        filepath (str): The path to the pickled dataset.

    Returns:
        pd.DataFrame | None: The loaded dataset as a pandas DataFrame.
    """

    try:
        data = pd.read_pickle(filepath)
        # print(f'Dataset loaded from pickle file: {filepath}.')
        return data

    except FileNotFoundError:
        # print('Pickle file not found.')
        return None


def load_merge_pickled_datasets(filepath_list: list) -> pd.DataFrame | None:
    """Load and merge sequentially datasets from pickle files.

    Args:
        filepath_list (list): A list of file paths to the pickled datasets.

    Returns:
        pd.DataFrame | None: The merged dataset as a pandas DataFrame.

    Raises:
        FileNotFoundError: If a pickle file is not found.
    """

    try:
        df_list = []

        # Load processed dataset
        if DATA_PROCESSED_PKL in filepath_list:
            columns = ['name', 'brand', 'reviews.rating', 'reviews.numHelpful', 'review']
            data_processed = pd.read_pickle(DATA_PROCESSED_PKL)[columns]
            df_list.append(data_processed)

        # Load clustered dataset
        if DATA_CLUSTERED_PKL in filepath_list:
            columns = ['review', 'clusterCategories']
            data_clustered = pd.read_pickle(DATA_CLUSTERED_PKL)[columns]
            df_list.append(data_clustered)

        # Load sentiment analysis dataset
        if DATA_SA_PKL in filepath_list:
            columns = ['review', 'reviews.sentiment', 'reviews.ft', 'reviews.ft.sentiment']
            data_sa = pd.read_pickle(DATA_SA_PKL)[columns]
            df_list.append(data_sa)

        # Merge the DataFrames sequentially
        data = reduce(lambda left, right: pd.merge(left, right, on='review', how='outer'), df_list)
        # print(f'Datasets loaded from pickle files:\n{filepath_list}.')
        return data

    except FileNotFoundError:
        # print(f'Pickle file not found.')
        return None


def print_random_product_sheet(data: pd.DataFrame) -> None:
    """Print a random product sheet from the dataset.

    Args:
        data (pd.DataFrame): The dataset to print a random product sheet from.
    """
    row = random.randint(0, data.shape[0])
    try:
        for col in data.columns:
            print(SEP)
            print(f'[{col}] {data[col][row]}')

    except KeyError:
        print(f'Row {row} not found in the dataset.')


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


def get_categories_from_dataset() -> list:
    """Extract unique categories from the scored dataset.

    Returns:
        list: The list of unique categories.
    """

    data_scored = load_pickled_dataset(DATA_SCORED_PKL)
    return list(data_scored['clusterCategories'].unique())


def get_top_products_per_category(category_name: str, top_n: int = 3) -> list:
    """Extract the top n products from a specific category.

    Args:
        category_name (str): The category to extract products from.
        top_n (int, optional): Extract top n products. Defaults to 3.

    Returns:
        list: The list of top n products from the category in descending order (best rating first)
    """
    # load the scored dataset
    data_scored = load_pickled_dataset(DATA_SCORED_PKL)
    # subset the category
    category_subset = data_scored[data_scored['clusterCategories'] == category_name]
    # select the top n products
    top_products = category_subset.head(top_n)
    # return names
    return top_products['name'].to_list()


def get_top_products_all_categories(data_scored: pd.DataFrame, top_n: int = 3) -> dict:
    """Extract the top n products from all categories and store them in a dict.

    Args:
        data_scored (pd.DataFrame): The dataset of scored products.
        top_n (int, optional): Extract top n products. Defaults to 3.

    Returns:
        dict: A dictionary of top n products per category.
    """

    categories = get_categories_from_dataset(data_scored)
    top_products = {}

    for cat in categories:
        top_products[cat] = get_top_products_per_category(data_scored, cat, top_n)

    return top_products


def get_random_product_review(subset: pd.DataFrame, min_word_count=10) -> str:
    """Returns a cleaned random product review from a subset of the dataset.

    Args:
        subset (pd.DataFrame): The subset of the dataset to extract reviews from.
        min_word_count (int, optional): Min review length in words. Defaults to 10.

    Returns:
        str: A random product review.
    """

    while True:
        # get a random review
        new_review = subset['review'].sample(random_state=random.randint(1, len(subset['review']))).values[0]
        # check if it has enough words
        if len(new_review.split()) >= min_word_count:
            # clean the review (forgotten steps in preprocessing)
            # remove line breaks
            new_review = new_review.replace('\n', ' ')
            # remove multiple dots
            new_review = re.sub(r'\.{2,}', '.', new_review)
            # remove multiple spaces
            new_review = re.sub(r'\s{2,}', ' ', new_review)
            # lowercase letters
            new_review = new_review.lower()
            return new_review


def get_product_reviews_per_sentiment(data_scored: pd.DataFrame, name: str, category: str, sentiment: str, n: int = 3) -> list:
    """Extract n reviews for a specific product, given a product name, category and sentiment.

    Args:
        data_scored (pd.DataFrame): The dataset of scored products.
        name (str): The product to extract reviews for.
        category (str): The category of the product.
        sentiment (str): The sentiment of the reviews to extract (positive, negative, neutral).
        n (int, optional): Number of reviews to extract. Defaults to 3.

    Returns:
        list: The list of reviews for the product.
    """

    # subset the data
    subset = data_scored[(data_scored['name'] == name) & (data_scored['clusterCategories'] == category) & (data_scored['reviews.sentiment'] == sentiment)]

    # get random reviews
    reviews = []
    while len(reviews) < n:
        new_review = get_random_product_review(subset)
        # append to list if not already in to avoid duplicates
        if new_review not in reviews:
            reviews.append(new_review)

    return reviews


def get_sentiment_ratio(data: pd.DataFrame) -> list:
    """Calculate the ratio of positive, neutral and negative reviews in the dataset.

    Args:
        data (pd.DataFrame): The dataset to calculate the sentiment ratio for.

    Returns:
        list: The ratio of positive, neutral and negative reviews in the dataset summing to 1.
    """

    total_reviews = data.shape[0]
    # If no reviews are found, return [0, 0, 0] to avoid division by zero
    if total_reviews == 0:
        return [0, 0, 0]

    # Count occurrences of each sentiment
    positive_count = data['reviews.ft.sentiment'].value_counts().get('positive', 0)
    neutral_count = data['reviews.ft.sentiment'].value_counts().get('neutral', 0)
    negative_count = data['reviews.ft.sentiment'].value_counts().get('negative', 0)

    # Calculate the ratio for each sentiment, rounded to 2 decimals
    positive_ratio = round(positive_count / total_reviews, 2)
    neutral_ratio = round(neutral_count / total_reviews, 2)
    negative_ratio = round(negative_count / total_reviews, 2)

    return [positive_ratio, neutral_ratio, negative_ratio]


def sample_product_reviews(name: str, category: str, n: int = 10) -> list:
    """Sample n reviews of a product, mixing positive, negative and neutral reviews based on score.
    The reviews are concatenated in a list and shuffled to avoid any bias.

    Args:
        name (str): The product to extract reviews for.
        category (str): The category of the product.
        n (int, optional): Number of reviews to extract. Defaults to 10.

    Returns:
        list: The list of sampled reviews for the product.
    """

    # load dataset with clustered categories and sentiment analysis
    data = load_merge_pickled_datasets([DATA_PROCESSED_PKL, DATA_CLUSTERED_PKL, DATA_SA_PKL])

    # subset the data for the product and category
    subset = data[(data['clusterCategories'] == category) & (data['name'] == name)]

    # get sentiment ratio for this product
    sentiment_ratio = get_sentiment_ratio(subset)

    # get the number of reviews per sentiment, at least 1 for each
    negative_n = max(1, int(n * sentiment_ratio[2]))
    neutral_n = max(1, int(n * sentiment_ratio[1]))
    # positive_n = max(1, int(n * sentiment_ratio[0]))
    positive_n = n - negative_n - neutral_n  # to sum up to n reviews

    # get reviews for each sentiment
    positive_reviews = get_product_reviews_per_sentiment(data, name, category, 'positive', positive_n)
    neutral_reviews = get_product_reviews_per_sentiment(data, name, category, 'neutral', neutral_n)
    negative_reviews = get_product_reviews_per_sentiment(data, name, category, 'negative', negative_n)

    # merge the reviews and shuffle them to avoid any bias
    all_reviews = positive_reviews + neutral_reviews + negative_reviews
    random.shuffle(all_reviews)

    return all_reviews
