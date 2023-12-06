from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from data_preprocessor import DataPreprocessor

from collections import Counter
import math

import numpy as np
from scipy.sparse import csr_matrix


class RelatedDocumentsRetrieval:
    def __init__(self, document_titles, documents):
        self.document_titles: List[str] = document_titles
        self.documents: List[str] = documents
        self.preprocessor = DataPreprocessor()
        self.vectorizer = TfidfVectorizer()

        self.all_words = set(word for doc in self.documents for word in doc.split())
        self._tfidf_matrix = self._calculate_tfidf()

    def preprocess_documents(self):
        """
        Preprocess all documents in the collection.
        """
        preprocessed_documents = [self.preprocessor.preprocess_text(doc) for doc in self.documents]
        return preprocessed_documents

    def vectorize_documents(self, preprocessed_documents):
        """
        Convert preprocessed documents into TF-IDF vectors.
        """
        preprocessed_documents = self.documents
        tfidf_matrix = self.vectorizer.fit_transform(preprocessed_documents)
        self._tfidf_matrix = self._calculate_tfidf()
        _tfidf_matrix = self._vectorize_documents()

        return _tfidf_matrix

    def retrieve_similar_documents(self, query_document, by_title="", num_results=5, is_query_preprocessed=False):
        """
        Retrieve similar documents to the given query document.
        """
        # Preprocess the query document
        preprocessed_query = query_document
        if not is_query_preprocessed:
            preprocessed_query = self.preprocessor.preprocess_text(query_document)
        # Vectorize the query document
        # query_vector = self.vectorizer.transform([preprocessed_query])
        query_vector = self._vectorize_query_document(preprocessed_query)
        # Calculate cosine similarity between the query and all documents
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        # Get indices of top similar documents
        similar_indices = similarities.argsort()[:-num_results-1:-1]
        # Retrieve and return the similar documents
        similar_documents = [self.documents[i] for i in similar_indices if not self.document_titles[i] == by_title]
        similar_documents_titles = [self.document_titles[i] for i in similar_indices if not self.document_titles[i] == by_title]
        return similar_documents_titles, similar_documents

    def _calculate_tfidf(self):
        """
        Multiply TF and IDF for each word in each document.
        :return:
        """
        tfidf_matrix = []

        idf = self._calculate_idf()

        for document in self.documents:
            tf = self._calculate_tf(document)
            tfidf = {word: tf[word] * idf[word] for word in tf}
            tfidf_matrix.append(tfidf)

        self._tfidf_matrix = tfidf_matrix
        return tfidf_matrix

    def _calculate_tf(self, document):
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

    def _calculate_idf(self):
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

    def _vectorize_documents(self):
        """
        Convert TF-IDF scores into vectors.
        :return:
        """
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

    def _vectorize_query_document(self, query_document):
        tf = self._calculate_tf(query_document)
        idf = self._calculate_idf()

        tfidf = {word: tf[word] * idf[word] for word in tf}
        vector = [tfidf.get(word, 0) for word in self.all_words]

        return vector


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
