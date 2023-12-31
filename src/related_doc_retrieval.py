import os
import time
from typing import List, Set, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from data_preprocessor import DataPreprocessor
from collections import Counter
import math
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from src import variables, utils


class RelatedDocumentsRetrieval:
    def __init__(self, document_titles, documents, use_own_vectorizer=True, use_own_cosine_similarity=True):
        self.document_titles: List[str] = document_titles
        self.documents: List[str] = documents

        if use_own_vectorizer:
            self.vectorizer: OwnTfidfVectorizer = OwnTfidfVectorizer()
        else:
            self.vectorizer: TfidfVectorizer = TfidfVectorizer()
        if use_own_cosine_similarity:
            self.cosine_similarity = self.cosineSimularity
        else:
            self.cosine_similarity = sklearn_cosine_similarity

        self.preprocessor: DataPreprocessor = DataPreprocessor()
        self._tfidf_matrix: csr_matrix = None

    def preprocess_documents(self):
        """
        Preprocess all documents in the collection.
        """
        preprocessed_documents = [self.preprocessor.preprocess_text(doc) for doc in self.documents]
        return preprocessed_documents

    def initialize_tf_idf_matrix(self, documents, force_rerun=False):
        """
        Initialize the TF-IDF matrix.
        :param documents: list of documents to vectorize.
        :param force_rerun: force rerun the vectorization process without saving the tfidf matrix to file.
        :return:
        """
        self._tfidf_matrix = None
        if isinstance(self.vectorizer, TfidfVectorizer): # if using sklearn's vectorizer, need to use their fit_transform
            tfidf_matrix = self.vectorizer.fit_transform(documents)
            self._tfidf_matrix = tfidf_matrix
        else: # if using our own vectorizer
            if force_rerun:
                self._tfidf_matrix = self.vectorize_documents()
            # if the tfidf matrix is already stored in a file, load it.
            # Otherwise, vectorize the documents and store the matrix in a file.
            elif not os.path.isfile(variables.tfidf_matrix_csv_path):
                self._tfidf_matrix = self.vectorize_documents()
                utils.store_tfidf_matrix(self._tfidf_matrix)
            else:
                self._tfidf_matrix = utils.load_tfidf_matrix()
                self.vectorizer.fit_transform(documents, self._tfidf_matrix)

    def vectorize_documents(self):
        """
        Convert preprocessed documents into TF-IDF vectors.
        """
        preprocessed_documents = self.documents
        tfidf_matrix: csr_matrix = self.vectorizer.fit_transform(preprocessed_documents)
        return tfidf_matrix

    def retrieve_similar_documents(self, query_document, query_title="", num_results=5, is_query_preprocessed=False):
        """
        Retrieve similar documents to the given query document.
        """
        tfidf_matrix = self._tfidf_matrix

        # Preprocess the query document
        preprocessed_query = query_document
        if not is_query_preprocessed:
            preprocessed_query = self.preprocessor.preprocess_text(query_document)

        # Vectorize the query document
        query_vector = self.vectorizer.transform([preprocessed_query])

        similarities = None
        if self.cosine_similarity == self.cosineSimularity:
            similarities = self.cosine_similarity(query_vector[0, :], tfidf_matrix)
        else:
            similarities = sklearn_cosine_similarity(query_vector, tfidf_matrix).flatten()

        # sorted indices of the most similar documents
        similar_indices_sorted = similarities.argsort()[::-1]
        num_results_most_similar = []
        idx = 0
        # get the most similar documents
        while len(num_results_most_similar) < num_results and idx < len(similar_indices_sorted):
            # if the document is not the query document, add it to the list of most similar documents
            if not self.document_titles[similar_indices_sorted[idx]] == query_title:
                num_results_most_similar.append(similar_indices_sorted[idx])

            idx += 1
        # get the scores for the similar indices
        similar_scores = similarities[num_results_most_similar]

        # Retrieve and return the similar documents
        similar_documents = [self.documents[i] for i in num_results_most_similar if not self.document_titles[i] == query_title]
        similar_documents_titles = [self.document_titles[i] for i in num_results_most_similar if not self.document_titles[i] == query_title]
        return similar_documents_titles, similar_documents, similar_scores

    @staticmethod
    def l2_norm(vector, axis=0):
        """
        Calculate L2 norm of a vector.
        :param vector: vector
        :return: L2 norm of the vector
        """
        from scipy.sparse.linalg import norm
        return norm(vector, 2, axis=axis)

    @staticmethod
    def l2_norm_original(vector, axis=0):
        """
        Calculate L2 norm of a vector.
        :param vector: vector
        :return: L2 norm of the vector
        """
        sq = np.square(vector)
        s = np.sum(sq, axis=axis)
        sqrt = np.sqrt(s)
        return sqrt

    @staticmethod
    def cosineSimilarityOriginal(x:csr_matrix, y:csr_matrix):
        """
        Calculate cosine similarity between two vectors.
        :param x: query vector in an array
        :param y: document vector in an array
        :return: cosine similarity between x and each row in the document matrix y

        THIS IS THE ORIGINAL IMPLEMENTATION USING ARRAYS, THIS IS NOT USED ANYMORE BECAUSE OF MEMORY ISSUES.
        cosineSimularity IS USED INSTEAD.
        """
        dot_product = np.dot(y, x)
        magnitude_y = RelatedDocumentsRetrieval.l2_norm_original(y, axis=1)
        magnitude_x = RelatedDocumentsRetrieval.l2_norm_original(x)
        similarity = dot_product / ( magnitude_x * magnitude_y )
        return similarity

    @staticmethod
    def cosineSimularity(x: csr_matrix, y: csr_matrix):
        """
        Calculate cosine similarity between two vectors.
        :param x: query vector in an array
        :param y: document vector in an array
        :return: cosine similarity between x and each row in the document matrix y
        """

        # Compute the dot product between x and y
        dot_product = y.dot(x.T)
        # square each element in x manually as x.power(2) does not work
        x = x.multiply(x)
        y = y.multiply(y)

        # Compute the L2 norms (magnitudes) of x and y
        magnitude_x = np.sqrt(np.sum(x))
        magnitude_y = np.sqrt(np.sum(y, axis=1))

        # Compute the cosine similarity
        cosine_similarity = dot_product / (magnitude_x * magnitude_y)
        returnval = cosine_similarity.transpose()
        returnval = returnval.toarray()[0]
        return returnval


class OwnTfidfVectorizer:

    def __init__(self):
        self.documents: List[str] = []
        self.document_titles: List[str] = []
        # if document_titles is not None:
        #     self.document_titles = document_titles

        self.all_words: List[str] = list()
        self._tfidf_matrix: pd.DataFrame = None

    def fit_transform(self, documents: List[str], tfidf_matrix=None):
        """
        Fit the vectorizer to the documents and transform the documents to a tfidf matrix.
        :param documents: list of documents
        :param tfidf_matrix: tfidf matrix
        :return: tfidf matrix: csr_matrix
        """
        self.documents = documents
        self.all_words = self.get_all_words()
        if tfidf_matrix is None:
            self._tfidf_matrix = self.vectorize_documents()
        else:
            self._tfidf_matrix = tfidf_matrix
        return self._tfidf_matrix

    def transform(self, query_document: List[str]):
        """
        Transform a query document to a tfidf matrix.
        :param query_document: query document as a single element of a list
        :return: query tfidf matrix: csr_matrix
        """
        query_document = query_document[0] # for compatibility with sklearn
        tf: Dict[str, float] = self.calculate_tf(query_document)
        idf: Dict[str, float] = self.calculate_idf()

        # multiply tf and idf for each word in the query document if the word is in the idf
        tfidf: Dict[str, float] = {word: tf[word] * idf[word] for word in tf if word in idf}

        # create a single row csr matrix
        tfidf_matrix = csr_matrix((1, len(self.all_words)))

        for word in tfidf:
            tfidf_matrix[0, self.all_words.index(word)] = tfidf[word]

        return tfidf_matrix

    def get_all_words(self):
        """
        Get all words in the collection of documents.
        """
        words: set = set()
        for doc in self.documents:
            for word in doc.split():
                words.add(word)
        # return an ordered list of words
        return list(sorted(words))


    def calculate_tfidf(self):
        """
        Multiply TF and IDF for each word in each document.
        :return: list of dictionaries with the TFIDF for each word in each document
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
        Convert TF-IDF scores into csr_matrix.
        :return: the csr_matrix of TF-IDF scores
        """
        self._tfidf_matrix = self.calculate_tfidf()

        print(f"Done calculating TF-IDF for {len(self._tfidf_matrix)} documents")
        rows, cols, data = [], [], []
        for i, tfidf in enumerate(self._tfidf_matrix):
            for j, word in enumerate(self.all_words):
                if word in tfidf:
                    rows.append(i)
                    cols.append(j)
                    data.append(tfidf[word])
            if i % 1000 == 0:
                print(f"Done with {i} documents")

        vectors = csr_matrix((data, (rows, cols)), shape=(len(self._tfidf_matrix), len(self.all_words)))
        return vectors


    def vectorize_documents_pandas(self):
        """
        Convert TF-IDF scores into pandas dataframe.
        :return: pandas dataframe
        """
        start = time.time()
        tfidf_list = self.calculate_tfidf()
        print(f"Done calculating TF-IDF for {len(tfidf_list)} documents in {time.time() - start} seconds")
        start = time.time()
        self._tfidf_matrix = pd.DataFrame(tfidf_list, columns=self.all_words).fillna(0.0)
        print(f"Done converting TF-IDF to pandas dataframe in {time.time() - start} seconds")
        return self._tfidf_matrix


def testCosineSimularity():
    """
    Test cosine simularity
    :return:
    """
    from scipy.sparse import csr_matrix

    B = csr_matrix(np.array([[2,1,2],[3,2,9], [-1,2,-3]]))
    A = csr_matrix(np.array([3,4,2]))
    cosim = RelatedDocumentsRetrieval.cosineSimularity(A, B)
    expected =  [ 0.86657824,  0.67035541, -0.04962917]

    print("A:", A)
    print("B:", B)
    print("Expected:\t",expected)
    print("Actual:\t",cosim)
    for id, i in enumerate(cosim):
        assert abs(i - expected[id]) < 0.0000001


def testSimilarDocuments():
    """
    Test similar documents retrieval function on a small toy example
    :return:
    """
    l = ['data science is one of the most important fields of science',
     'this is one of the best data science courses',
     'data scientists analyze data']

    q = ['data scientists is analyze courses data space']

    document_titles = ["doc1", "doc2", "doc3"]
    documents = l

    retrieval_system = RelatedDocumentsRetrieval(document_titles, documents)
    retrieval_system.initialize_tf_idf_matrix(documents, force_rerun=True)

    similar_documents_titles, similar_documents, scores = retrieval_system.retrieve_similar_documents(q[0], "", 5, is_query_preprocessed=True)
    assert similar_documents_titles == ['doc3', 'doc2', 'doc1']


if __name__ == '__main__':
    testCosineSimularity()
    testSimilarDocuments()
