from test import calculatePrecisionAndRecallfrom utils import *from related_doc_retrieval import *from gui import run_guifrom src import variablesif __name__ == '__main__':    doc_dict = getDocDict(filepath_video_games=variables.filepath_video_games, csv_doc_dict=variables.csv_doc_dict)    documents = list(doc_dict.values())    document_titles = list(doc_dict.keys())    retrieval_system = RelatedDocumentsRetrieval(document_titles, documents)    # check if tfidf_matrix.csv exists    retrieval_system._tfidf_matrix = None    if not os.path.isfile(variables.tfidf_matrix_csv_path):        retrieval_system._tfidf_matrix = retrieval_system.vectorize_documents()        store_tfidf_matrix(retrieval_system._tfidf_matrix)    else:        retrieval_system._tfidf_matrix = load_tfidf_matrix()        retrieval_system.own_vectorizer.fit_transform(documents, retrieval_system._tfidf_matrix)    run_gui(doc_dict, retrieval_system)