from utils import *from gui import run_guifrom src import variablesfrom src.inverted_index import SPIMIif __name__ == '__main__':    doc_dict = getDocDict(filepath_video_games=variables.filepath_video_games, csv_doc_dict=variables.csv_doc_dict)    documents = list(doc_dict.values())    document_titles = list(doc_dict.keys())    retrieval_system: SPIMI = SPIMI(block_size_limit=10000, force_reindex=True, documents=list(doc_dict.values()), document_titles=document_titles)    run_gui(doc_dict, retrieval_system)