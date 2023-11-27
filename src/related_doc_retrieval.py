from data_preprocessor import DataPreprocessor
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class RelatedDocumentsRetrieval:
    def __init__(self, documents):
        self.documents: List[str] = documents
        self.preprocessor = DataPreprocessor()
        self.vectorizer = TfidfVectorizer()

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
        tfidf_matrix = self.vectorizer.fit_transform(preprocessed_documents)
        return tfidf_matrix

    def retrieve_similar_documents(self, query_document, num_results=5):
        """
        Retrieve similar documents to the given query document.
        """
        # Preprocess the query document
        preprocessed_query = self.preprocessor.preprocess_text(query_document)
        # Vectorize the query document
        query_vector = self.vectorizer.transform([preprocessed_query])
        # Calculate cosine similarity between the query and all documents
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        # Get indices of top similar documents
        similar_indices = similarities.argsort()[:-num_results-1:-1]
        # Retrieve and return the similar documents
        similar_documents = [self.documents[i] for i in similar_indices]
        return similar_documents


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
    retrieval_system.tfidf_matrix = retrieval_system.vectorize_documents(preprocessed_documents)

    query_document = "Players compete in various game modes."
    similar_documents = retrieval_system.retrieve_similar_documents(query_document)

    print("Query Document:", query_document)
    print("Similar Documents:")
    for i, doc in enumerate(similar_documents, 1):
        print(f"{i}. {doc}")
