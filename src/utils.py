import csv
import re
import sys
import ast
from data_preprocessor import DataPreprocessor


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
        reader = csv.DictReader(file)

        maxInt = sys.maxsize
        while True:
            # decrease the maxInt value by factor 10
            # as long as the OverflowError occurs.
            try:
                csv.field_size_limit(maxInt)
                print("csv.field_size_limit = ", maxInt)
                break
            except OverflowError:
                maxInt = int(maxInt / 10)

        # Iterate over each row in the CSV file
        for row in reader:
            title = row['Title']
            sections_str = row['Sections']

            # read the sections_str literal to a list
            sections = ast.literal_eval(sections_str)

            doc_dict[title] = sections

    return doc_dict

def getDocDict(filepath_video_games, csv_doc_dict):
    if os.path.exists(csv_doc_dict):
        # read the dictionary from the csv file into a dataframe
        df = pd.read_csv(csv_doc_dict, encoding='utf-8')
        # convert the dataframe to a dictionary
        doc_dict = df.set_index('Title').T.to_dict('list')
        return doc_dict
    else:
        # read the dataset file into a dictionary
        doc_dict = read_dataset_file(filepath_video_games)
        # read the dictionary into a panda dataframe
        df = pd.DataFrame.from_dict(doc_dict, orient='index', columns=['Text'])
        # write the dataframe to a csv file
        df.to_csv(csv_doc_dict, encoding='utf-8')
        return doc_dict

def write_dict_to_csv(dict, columns, filename):
    """
    Write a dictionary to a CSV file.
    :param dict: dictionary to write
    :param columns: column names
    :param filename: file name
    :return:
    """
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        for key in dict.keys():
            row = {columns[0]:key}
            for column in columns[1:]:
                row[column] = dict[key]
            try:
                writer.writerow(row)
            except:
                print(row)


def read_dict_from_csv(filename, columns):
    """
    Read a dictionary from a CSV file.
    :param filename:
    :param columns:
    :return: dictionary
    """
    with open(filename, 'r', newline='', encoding='utf-8') as csvfile:
        return {row[columns[0]]:row[columns[1]] for row in csv.DictReader(csvfile)}
