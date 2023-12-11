import time
from typing import List, Set, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_preprocessor import DataPreprocessor

from collections import Counter
import math

import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from IPython.display import display


class RelatedDocumentsRetrieval:
    def __init__(self, document_titles, documents, use_own_vectorizer=True):
        self.document_titles: List[str] = document_titles
        self.documents: List[str] = documents

        self.preprocessor: DataPreprocessor = DataPreprocessor()

        self.vectorizer: TfidfVectorizer = TfidfVectorizer()
        self.own_vectorizer: OwnTfidfVectorizer = OwnTfidfVectorizer(document_titles)

    def preprocess_documents(self):
        """
        Preprocess all documents in the collection.
        """
        preprocessed_documents = [self.preprocessor.preprocess_text(doc) for doc in self.documents]
        return preprocessed_documents

    def vectorize_documents(self):
        """
        Convert preprocessed documents into TF-IDF vectors.
        """
        preprocessed_documents = self.documents
        # tfidf_matrix = self.vectorizer.fit_transform(preprocessed_documents)
        _tfidf_matrix = self.own_vectorizer.fit_transform(preprocessed_documents)

        # display(_tfidf_matrix)
        # # _tfidf_matrix.to_csv('output/reference_dataset.csv', index=False)
        # _tfidf_matrix.to_csv('output/cur_dataset.csv', index=False)
        # cur_df = pd.read_csv('output/cur_dataset.csv')
        # reference_df = pd.read_csv('output/reference_dataset.csv')
        # # Sort columns for both DataFrames
        # cur_df = cur_df.sort_index(axis=1)
        # reference_df = reference_df.sort_index(axis=1)
        # if cur_df.equals(reference_df):
        #     print("The two matrices are equal")
        # else:
        #     print("The two matrices are not equal")

        return _tfidf_matrix

    def retrieve_similar_documents(self, query_document, by_title="", num_results=5, is_query_preprocessed=False):
        """
        Retrieve similar documents to the given query document.
        """
        tfidf_matrix = self.vectorize_documents()

        # Preprocess the query document
        preprocessed_query = query_document
        if not is_query_preprocessed:
            preprocessed_query = self.preprocessor.preprocess_text(query_document)

        # Vectorize the query document
        _query_vector = self.vectorizer.transform([preprocessed_query])
        query_vector = self.own_vectorizer.transform([preprocessed_query])

        # Calculate cosine similarity between the query and all documents
        similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

        # Get indices of top similar documents
        similar_indices = similarities.argsort()[:-num_results-1:-1]

        # Retrieve and return the similar documents
        similar_documents = [self.documents[i] for i in similar_indices if not self.document_titles[i] == by_title]
        similar_documents_titles = [self.document_titles[i] for i in similar_indices if not self.document_titles[i] == by_title]
        return similar_documents_titles, similar_documents


class OwnTfidfVectorizer:

    def __init__(self, document_titles: List[str] = None):
        self.documents: List[str] = []
        self.document_titles: List[str] = []
        if document_titles is not None:
            self.document_titles = document_titles

        self.all_words: List[str] = list()
        self._tfidf_matrix: pd.DataFrame = None

    def fit_transform(self, documents: List[str]):
        self.documents = documents
        self.all_words = self.get_all_words()
        self._tfidf_matrix = self.vectorize_documents()
        return self._tfidf_matrix

    def transform(self, query_document: List[str]):
        query_document = query_document[0] # for compatibility with sklearn
        tf: Dict[str, float] = self.calculate_tf(query_document)
        idf: Dict[str, float] = self.calculate_idf()

        tfidf: Dict[str, float] = {word: tf[word] * idf[word] for word in tf}
        vector = [tfidf.get(word, 0) for word in self.all_words]

        # make pandas dataframe
        df = pd.DataFrame([vector], columns=self.all_words)
        return df

    def get_all_words(self):
        """
        Get all words in the collection of documents.
        """
        return list(set(word for doc in self.documents for word in doc.split()))

    def calculate_tfidf(self):
        """
        Multiply TF and IDF for each word in each document.
        :return:
        """
        idf: Dict[str, float] = self.calculate_idf()

        tf_idf_list = []

        for idx, document in enumerate(self.documents):

            tf: Dict[str, float] = self.calculate_tf(document)
            tfidf: Dict[str, float] = {word: tf[word] * idf[word] for word in tf}

            tf_idf_list.append(tfidf)

        return tf_idf_list

    def calculate_tf(self, document):
        """
        Tokenize each document to obtain a list of words.
        Count the frequency of each word in the document
        :param document:
        :return:
        """
        words = document.split()
        word_counts = Counter(words)
        tf: Dict[str, float] = {word: count / len(words) for word, count in word_counts.items()}
        return tf

    def calculate_idf(self):
        """
        Count the number of documents containing each word.
        Calculate IDF for each word.
        :return:
        """
        word_doc_count = {}
        total_documents = len(self.documents)

        for document in self.documents:
            words = set(document.split())
            for word in words:
                word_doc_count[word] = word_doc_count.get(word, 0) + 1

        idf: Dict[str, float] = {word: math.log(total_documents / (count + 1)) for word, count in word_doc_count.items()}

        return idf

    def vectorize_documents(self):
        """

        :return:
        """
        # self._tfidf_matrix = self.calculate_tfidf()
        # return self._tfidf_matrix
        start = time.time()
        tfidf_list = self.calculate_tfidf()
        print(f"Done calculating TF-IDF for {len(tfidf_list)} documents in {time.time() - start} seconds")
        start = time.time()
        self._tfidf_matrix = pd.DataFrame(tfidf_list, columns=self.all_words).fillna(0.0)
        print(f"Done converting TF-IDF to pandas dataframe in {time.time() - start} seconds")
        # start = time.time()
        # self._tfidf_matrix.fillna(0.0, inplace=True)
        # print(f"Done filling NaN values in {time.time() - start} seconds")
        return self._tfidf_matrix


if __name__ == '__main__':
    documents = [
        "In the game, players have the choice to compete across any of the game modes.",
        "Gameplay involves teams of five players on indoor courts.",
        "The game was created by the team that later worked on Aero Fighters franchise.",
        "Power Spikes II received mixed reception from critics.",
        "Development involved collaboration with various designers and composers."
    ]

    retrieval_system = RelatedDocumentsRetrieval(documents)
    preprocessed_documents = retrieval_system.preprocess_documents()
    retrieval_system._tfidf_matrix = retrieval_system.vectorize_documents(preprocessed_documents)

    query_document = "Players compete in various game modes."
    similar_documents = retrieval_system.retrieve_similar_documents(query_document)

    print("Query Document:", query_document)
    print("Similar Documents:")
    for i, doc in enumerate(similar_documents, 1):
        print(f"{i}. {doc}")
