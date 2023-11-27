from utils import *from data_preprocessor import *import osfilepath_video_games = 'input/video_games.txt'csv_doc_dict = './output/processed_docs.csv.gz'if __name__ == '__main__':    doc_dict = getDocDict(filepath_video_games= filepath_video_games, csv_doc_dict= csv_doc_dict)    # Using a loop to print the first 20 key-value pairs    count = 0    for key, value in doc_dict.items():        print(f"{key}: {value}")        count += 1        if count == 20:            break    # Preprocess the text    input_text = "In the game, players have the choice to compete across any of the game modes."    preprocessor = DataPreprocessor()    preprocessed_text = preprocessor.preprocess_text(input_text)    print("Original Text:", input_text)    print("Preprocessed Text:", preprocessed_text)