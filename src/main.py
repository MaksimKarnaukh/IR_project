from utils import *from data_preprocessor import *import osfilepath_video_games = 'input/video_games.txt'csv_doc_dict = './output/test.csv'if __name__ == '__main__':    # dict = {"a":"aaaa", "b": "bbb","c":"cccccc"}    # writeDictToCSV(dict,["Title", "Text"], csv_doc_dict)    doc_dict = {}    if os.path.exists(csv_doc_dict):        doc_dict = readDictFromCSV(csv_doc_dict)    else:        # read the dataset file into a dictionary        doc_dict = read_dataset_file(filepath_video_games)        dict_sections_to_string(doc_dict)        # write the dictionary to a CSV file        writeDictToCSV(doc_dict, ["Title", "Text"], csv_doc_dict)    # Using a loop to print the first 20 key-value pairs    count = 0    for key, value in doc_dict.items():        print(f"{key}: {value}")        count += 1        if count == 20:            break    # Preprocess the text    input_text = "In the game, players have the choice to compete across any of the game modes."    preprocessor = DataPreprocessor()    preprocessed_text = preprocessor.preprocess_text(input_text)    print("Original Text:", input_text)    print("Preprocessed Text:", preprocessed_text)