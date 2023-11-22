from utils import *

if __name__ == '__main__':

    print('Hello World!')

    d = read_gt('input/gt')

    # print(d["Assassin's Creed IV: Black Flag"])

    import csv
    import re


    def parse_sections(sections_str):
        # Use regular expression to extract sections inside square brackets
        print(sections_str)
        sections = re.findall(r'\[(.*?)\]', sections_str)
        return sections


    file_path = 'input/video_games.txt'

    # Open the CSV file
    with open(file_path, 'r', encoding='utf-8') as file:
        # Create a CSV reader
        reader = csv.DictReader(file)

        idx = 0
        # Iterate over each row in the CSV file
        for row in reader:
            title = row['Title']
            sections_str = row['Sections']

            # Parse sections using the defined function
            sections = parse_sections(sections_str)

            # Print the results
            print(f'Title: {title}')
            print(f'Sections: {sections}')
            print('---')

            idx += 1

            if idx == 10:
                break