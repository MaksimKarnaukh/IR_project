import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk

class DataPreprocessor:
    def __init__(self, language='english'):
        self.stop_words = set(stopwords.words(language))
        self.stemmer = PorterStemmer()

    def tokenize(self, text):
        """
        Tokenize the input text. Returns tokens.
        """
        return word_tokenize(text)

    def remove_punctuation(self, text):
        """
        Remove punctuation from the input text.
        """
        return text.translate(str.maketrans("", "", string.punctuation)) # https://www.w3schools.com/python/ref_string_maketrans.asp

    def remove_stop_words(self, tokens):
        """
        Remove stop words from the list of tokens. (https://www.geeksforgeeks.org/removing-stop-words-nltk-python/)
        """
        return [word for word in tokens if word.lower() not in self.stop_words]

    def stem_tokens(self, tokens):
        """
        Stem the list of tokens using the Porter stemming algorithm.
        """
        return [self.stemmer.stem(word) for word in tokens]

    def preprocess_text(self, text):
        """
        Apply the entire data preprocessing pipeline to the input text.
        The steps are: lowercasing, punctuation removal, tokenization, stop word removal, stemming.
        """
        # Lowercasing
        text = text.lower()
        # Remove punctuation
        text = self.remove_punctuation(text)
        # Tokenization
        tokens = self.tokenize(text)
        # Remove stop words
        tokens = self.remove_stop_words(tokens)
        # Stemming
        tokens = self.stem_tokens(tokens)
        # Remove unicode characters
        tokens = self.remove_unicode(tokens)
        # Reassemble the preprocessed text
        preprocessed_text = " ".join(tokens)

        return preprocessed_text

    def check_nltk_data_downloaded(self):
        """
        Check if the required NLTK data is downloaded.
        """
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')


    def remove_unicode(self, tokens):
        """
        Remove all Unicode characters from a string.
        :param input_str: input string
        :return: string with only ASCII characters
        """
        # remove all non-ASCII characters from each token
        new_tokens = []
        for token in tokens:
            new_token = re.sub(r'[^\x00-\x7F]+', '', token)
            if new_token != '':
                new_tokens.append(new_token)
        return new_tokens