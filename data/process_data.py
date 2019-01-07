# import packages
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine
import sys


def load_data(messages_filepath, categories_filepath):
    """Method to load 2 csv files and create 1 dataframe to be returned

    Args:
        message_filepath : path to the file containing messages
        categories_filepath : path to the file containing categories where
                                each of messages were classified

    Returns:
        df : merged dataframe containg messages and the labels
    """

    ### read in 2 files and merge them together
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id")

    return df




def clean_data(df):
    """Method to clean the dataframe and return it

    Args :
        df : dataframe to be cleaned

    Returns :
        df : cleaned dataframe
    """

    # extract a dataframe of the 36 individual category columns
    df_categories = df.categories.str.split(";", expand=True)

    # make new column names for each of category columns
    row = df_categories.iloc[0]
    df_category_colnames = row.str[:-2].tolist()

    # assign the new column names
    df_categories.columns = df_category_colnames

    # loop through each column of categories to extract 0 or 1 of each cell
    for column in df_categories:

        # set each value to be the last character of the strings
        # then convert column from string to numeric
        df_categories[column] = df_categories[column].str[-1].astype(int)

    # concatenate the original dataframe with the new `categories` dataframe
    df = df.join(df_categories)

    # drop the original 'categories' and 'child_alone' columns from `df`
    df.drop(axis=1, labels=["categories", "child_alone"], inplace=True)

    # drop duplicates
    df = df.drop_duplicates()

    # convert '2' in "related" column to '0'
    df.related = df.related.replace(2, 0)

    return df




def save_data(df, database_filepath):
    """Method to store the cleaned dataframe to a database file

    Args:
        df : dataframe which would be stored
        database_filepath : path to the file where the data would be stored
    """

    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df.to_sql('messages', engine, index=False, if_exists="replace")




def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)

        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
