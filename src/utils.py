import csv
import os
import re
import sys
import ast
import zipfile
from data_preprocessor import DataPreprocessor
from alive_progress import alive_it
import pandas as pd
import csv
import gzip

def read_gt(filename):
    """
    Read ground truth from a file.
    :param filename: file name.
    :return: deserialized data
    """
    import pickle
    with open(filename, 'rb') as binary_file:
        binary_data = binary_file.read()
    deserialized_data = pickle.loads(binary_data)
    return deserialized_data


def section_to_string(section):
    """
    Convert a section to a string.
    :param section: section to convert
    :return: string representation of the section
    """
    big_string = " ".join(section)
    return big_string


def sections_to_string(sections):
    """
    Convert a dictionary of sections to a string.
    :param doc_dict:
    :return:
    """
    preprocessor = DataPreprocessor()
    doc_str = ''
    for section in sections:
        doc_str += section_to_string(section)
    doc_str = preprocessor.preprocess_text(doc_str)
    return doc_str


def read_dataset_file(filepath_video_games):
    """
    Read the dataset file.
    :param filepath_video_games: path to the dataset file.
    :return: dictionary of title and sections.
    """
    doc_dict = {}
    preprocessor = DataPreprocessor()

    # Open the CSV file
    with open(filepath_video_games, 'r', encoding='utf-8') as file:

        # Create a CSV reader
        reader:csv.DictReader = csv.DictReader(file)

        maxInt = sys.maxsize
        while True:
            # decrease the maxInt value by factor 10
            # as long as the OverflowError occurs.
            try:
                csv.field_size_limit(maxInt)
                # print("csv.field_size_limit = ", maxInt)
                break
            except OverflowError:
                maxInt = int(maxInt / 10)
        # get the number of rows in the CSV file
        rows = list(reader)
        nr_rows = len(rows)
        # print the number of rows
        # print("Number of rows: ", nr_rows)
        # Iterate over each row in the CSV file
        for row in alive_it(rows, title='Reading dataset file'):
            title = row['Title']
            sections_str = row['Sections']

            # read the sections_str literal to a list
            sections = ast.literal_eval(sections_str)

            doc_dict[title] = sections_to_string(sections)


    return doc_dict

def zip_file(filepath, delete_original=True, output_filepath= None, output_filename=None):
    """
    Zip a file.
    :param filepath: path to the file to zip
    :param delete_original: delete the original file
    :param output_filepath: path to the output file
    :param output_filename: name of the output file
    :return:
    """
    if output_filepath is None:
        output_filepath = os.path.dirname(filepath)
    if output_filename is None:
        output_filename = os.path.basename(filepath) + '.zip'
    with zipfile.ZipFile(os.path.join(output_filepath, output_filename), 'w', zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(filepath, os.path.basename(filepath))
    if delete_original:
        os.remove(filepath)


def getDocDict(filepath_video_games, csv_doc_dict):
    if os.path.exists(csv_doc_dict):
        # read the dictionary from the csv file into a dataframe
        df = pd.read_csv(csv_doc_dict, encoding='utf-8')
        # convert the dataframe to a dictionary key:str =Title, value:str =Text
        doc_dict = df.set_index('Title').T.to_dict('records')[0]
        return doc_dict
    else:
        # read the dataset file into a dictionary
        doc_dict = read_dataset_file(filepath_video_games)
        # read the dictionary into a panda dataframe
        # make sure the name of the index column is 'Title' and the name of the second column is 'Text'
        df = pd.DataFrame.from_dict(doc_dict, orient='index', columns=['Text']).rename_axis("Title", axis=0)
        # write the dataframe to a csv file with colums 'Title' and 'Text'
        df.to_csv(csv_doc_dict, encoding='utf-8')
        return doc_dict


def write_dict_to_csv(dict, columns, csvfile):
    """
    Write a dictionary to a CSV file.
    :param dict: dictionary to write
    :param columns: column names
    :param csvfile: the csvfile
    :return:
    """
    writer = csv.DictWriter(csvfile, fieldnames=columns)
    writer.writeheader()
    for key in dict.keys():
        row = {columns[0]:key}
        for column in columns[1:]:
            row[column] = dict[key]
            writer.writerow(row)


def read_dict_from_csv(csvfile, columns):
    """
    Read a dictionary from a CSV file. The first column is the key, the second column is the value.
    :param columns:
    :param csvfile: the csvfile
    :return: dictionary
    """
    return {row[columns[0]]:row[columns[1]] for row in csv.DictReader(csvfile)}
