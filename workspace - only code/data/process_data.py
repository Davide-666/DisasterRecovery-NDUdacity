import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Reads two CSV files and imports the related data as pandas dataframe,
    then merges them into a single dataframe
    Args:
    messages_file_path str: Messages CSV file
    categories_file_path str: Categories CSV file
    Returns:
    merged_df pandas_dataframe: Dataframe obtained from merging the two input\
    data
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, how = 'left', on = ['id'])
    return df

def clean_data(df):
    """
    Cleans the df created by load_data()
    Args:
    df pandas_dataframe: Merged dataframe returned from load_data() function
    Returns:
    df pandas_dataframe: Cleaned data to be used by ML model
    """
    categories = df['categories'].str.split(';', expand = True)
    row = categories.iloc[0]
    category_colnames = row.transform(lambda x: x[:-2]).tolist()    
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].transform(lambda x: x[-1:])
        categories[column] = pd.to_numeric(categories[column])
    df.drop('categories', axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)
    df.drop_duplicates(inplace = True)
    df = df[df['related'] != 2]
    return df

def save_data(df, database_filename):
    """
    Saves cleaned data to an SQL database
    Args:
    df pandas_dataframe: Cleaned data returned from clean_data() function
    database_file_name str: File path of SQL Database into which the cleaned
                            data is to be saved
    Returns: None
    """
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace') 


def main():
    """
    Runs the ETL pipeline:
    - extracts messages and categories from csv files
	- cleans and pre-processes data
	- loads the clean dataframe to SQLite local database
	"""
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