from typing import List, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from data_preprocessor import DataPreprocessor

from collections import Counter
import math

import numpy as np
from scipy.sparse import csr_matrix


class RelatedDocumentsRetrieval:
    def __init__(self, document_titles, documents, use_own_vectorizer=True, use_own_cosine_similarity=True):
        self.document_titles: List[str] = document_titles
        self.documents: List[str] = documents

        if use_own_vectorizer:
            # todo remove
            self.vectorizer: TfidfVectorizer = TfidfVectorizer()
            self.vectorizer: OwnTfidfVectorizer = OwnTfidfVectorizer()
        else:
            self.vectorizer: TfidfVectorizer = TfidfVectorizer()
            # todo remove
            self.own_vectorizer: OwnTfidfVectorizer = OwnTfidfVectorizer()

        if use_own_cosine_similarity:
            self.cosine_similarity = self.calculate_cosine_simularities
        else:
            self.cosine_similarity = sklearn_cosine_similarity

        self.preprocessor: DataPreprocessor = DataPreprocessor()


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
        tfidf_matrix = self.vectorizer.fit_transform(preprocessed_documents)
        _tfidf_matrix = self.own_vectorizer.fit_transform(preprocessed_documents)
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
        # query_vector = self.vectorizer.transform([preprocessed_query])
        query_vector = self.own_vectorizer.transform([preprocessed_query])

        # Calculate cosine similarity between the query and all documents
        similarities = self.cosine_similarity(query_vector, tfidf_matrix).flatten()

        # Get indices of top similar documents
        similar_indices = similarities.argsort()[:-num_results-1:-1]

        # Retrieve and return the similar documents
        similar_documents = [self.documents[i] for i in similar_indices if not self.document_titles[i] == by_title]
        similar_documents_titles = [self.document_titles[i] for i in similar_indices if not self.document_titles[i] == by_title]
        return similar_documents_titles, similar_documents

    @staticmethod
    def l2_norm(vector, axis=0):
        """
        Calculate L2 norm of a vector.
        :param vector: vector
        :return: L2 norm of the vector
        """
        return np.sqrt(np.sum(np.square(vector), axis=axis))

    @staticmethod
    def cosineSimularity(x,y):
        """
        Calculate cosine similarity between two vectors.
        :param x: first vector in a panda series
        :param y: second vector in a panda dataframe
        :return: cosine similarity between x and y
        """
        return np.dot(y, x) / (RelatedDocumentsRetrieval.l2_norm(y, axis=1) * RelatedDocumentsRetrieval.l2_norm(x   ))


class OwnTfidfVectorizer:

    def __init__(self):
        self.documents = []
        self.all_words = []
        self._tfidf_matrix = None

    def fit_transform(self, documents):
        self.documents = documents
        self.all_words = self.get_all_words()
        self._tfidf_matrix = self.vectorize_documents()
        return self._tfidf_matrix

    def transform(self, query_document) -> List[float]:
        query_document = query_document[0] # for compatibility with sklearn
        tf = self.calculate_tf(query_document)
        idf = self.calculate_idf()

        tfidf = {word: tf[word] * idf[word] for word in tf}
        vector = [tfidf.get(word, 0) for word in self.all_words]

        return vector

    def get_all_words(self):
        """
        Get all words in the collection of documents.
        """
        return set(word for doc in self.documents for word in doc.split())

    def calculate_tfidf(self):
        """
        Multiply TF and IDF for each word in each document.
        :return:
        """
        tfidf_matrix = []

        idf = self.calculate_idf()

        for document in self.documents:
            tf = self.calculate_tf(document)
            tfidf = {word: tf[word] * idf[word] for word in tf}
            tfidf_matrix.append(tfidf)

        self._tfidf_matrix = tfidf_matrix
        return tfidf_matrix

    def calculate_tf(self, document):
        """
        Tokenize each document to obtain a list of words.
        Count the frequency of each word in the document
        :param document:
        :return:
        """
        words = document.split()
        word_counts = Counter(words)
        tf = {word: count / len(words) for word, count in word_counts.items()}
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

        idf = {word: math.log(total_documents / (count + 1)) for word, count in word_doc_count.items()}
        return idf

    def vectorize_documents(self):
        """
        Convert TF-IDF scores into csr_matrix.
        :return:
        """
        self._tfidf_matrix = self.calculate_tfidf()
        # vectors = []
        # for tfidf in self._tfidf_matrix:
        #     vector = [tfidf.get(word, 0) for word in self.all_words]
        #     vectors.append(vector)
        # return vectors
        rows, cols, data = [], [], []
        for i, tfidf in enumerate(self._tfidf_matrix):
            for j, word in enumerate(self.all_words):
                if word in tfidf:
                    rows.append(i)
                    cols.append(j)
                    data.append(tfidf[word])
        vectors = csr_matrix((data, (rows, cols)), shape=(len(self._tfidf_matrix), len(self.all_words)))
        return vectors

def testCosineSimularity():
    # define two lists or array
    # A = np.array([2, 1, 2, 3, 2, 9])
    # B = np.array([3, 4, 2, 4, 5, 5])
    # cosim = RelatedDocumentsRetrieval.cosineSimularity(A, B)
    # expected = 0.8188504723485274
    #
    # print("A:", A)
    # print("B:", B)
    # print("Expected:\t",expected)
    # print("Actual:\t",cosim)
    # assert abs(cosim - expected) < 0.0000000000000001
    B = np.array([[2,1,2],[3,2,9], [-1,2,-3]])
    A = np.array([3,4,2])
    cosim = RelatedDocumentsRetrieval.cosineSimularity(A, B)
    expected =  [ 0.86657824,  0.67035541, -0.04962917]

    print("A:", A)
    print("B:", B)
    print("Expected:\t",expected)
    print("Actual:\t",cosim)
    for id,i in enumerate(cosim):

        assert abs(i - expected[id]) < 0.0000001


if __name__ == '__main__':
    # define two lists or array
    A = np.array([[2, 1, 2], [3, 2, 9], [-1, 2, -3]])
    B = np.array([3, 4, 2])
    print("A:\n", A)
    print("B:\n", B)

    # compute cosine similarity
    cosine = np.dot(A, B) / (RelatedDocumentsRetrieval.l2_norm(A, axis=1) * RelatedDocumentsRetrieval.l2_norm(B))
    print("Cosine Similarity:\n", cosine)
    testCosineSimularity()
    # documents = [
    #     "In the game, players have the choice to compete across any of the game modes.",
    #     "Gameplay involves teams of five players on indoor courts.",
    #     "The game was created by the team that later worked on Aero Fighters franchise.",
    #     "Power Spikes II received mixed reception from critics.",
    #     "Development involved collaboration with various designers and composers."
    # ]
    #
    # retrieval_system = RelatedDocumentsRetrieval(documents)
    # preprocessed_documents = retrieval_system.preprocess_documents()
    # retrieval_system._tfidf_matrix = retrieval_system.vectorize_documents(preprocessed_documents)
    #
    # query_document = "Players compete in various game modes."
    # similar_documents = retrieval_system.retrieve_similar_documents(query_document)
    #
    # print("Query Document:", query_document)
    # print("Similar Documents:")
    # for i, doc in enumerate(similar_documents, 1):
    #     print(f"{i}. {doc}")
