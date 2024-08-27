import numpy as np

from src.inverted_index import SPIMI
from src.related_doc_retrieval import RelatedDocumentsRetrieval
from utils import *
import time
import variables


def calculate_confusion_matrix(expected,retrieved):
    # number of retrieved values that are also in expected (True positive)
    TP = len([ret for ret in retrieved if ret in expected])
    # number of retrieved values that aren't in expected (False positive)
    FP = len([ret for ret in retrieved if ret not in expected])
    # number of expected values that aren't retrieved (False negative)
    FN = len([ex for ex in expected if ex not in retrieved])

    return TP,FP,FN

def calculate_precision(expected, retrieved, k=0) -> float:
    """
    • P(recision) = TP/(TP+FP) how much correct of found
    :param expected: list of expected values (ground truth labels)
    :param retrieved: list of retrieved values (our algorithm)
    :param k: cut-off value (top k retrieved values)
    :return: tuple(precision, recall)
    """
    if k != 0:
        retrieved = retrieved[:k]
    TP, FP, FN = calculate_confusion_matrix(expected, retrieved)


    # precision calculation (by formula)
    precision = TP/(TP+FP) if TP+FP != 0 else 0

    return precision

def calculate_recall(expected, retrieved, k=0):
    """
    • R(ecall) = TP/(TP+FN) how much of correct
    :param expected: list of expected values (ground truth labels)
    :param retrieved: list of retrieved values (our algorithm)
    :param k: cut-off value (top k retrieved values)
    :return: tuple(precision, recall)
    """
    if k != 0:
        retrieved = retrieved[:k]
    TP, FP, FN = calculate_confusion_matrix(expected, retrieved)


    # recall calculation (by formula)
    recall = TP/(TP+FN) if TP+FN != 0 else 0

    return recall

def calculate_average_precision(expected, retrieved, k=0):
    if k == 0:
        k = len(retrieved)
    average_precision_at_k = 0
    for i in range(0, k):
        if retrieved[i] in expected:
            precision_at_k = calculate_precision(expected, retrieved, k=i+1)
            average_precision_at_k += precision_at_k
    average_precision_at_k /= k

    return average_precision_at_k

# def calculateF1score(precision, recall) -> float:
#     """
#     F1 score calculation (by formula)
#     F1 = 2*P*R/(P+R)
#     :param precision: P
#     :param recall: R
#     :return: F1 score
#     """
#     # F1 score calculation (by formula)
#     F1_score = (2*precision*recall) + (precision+recall)
#     return F1_score


def calculateKappa(expected_pairs: dict, retrieved_pairs: dict, nr_of_docs, relevance_threshold=0.7):
    """
    kappa >= 2/3 is good
    :param expected_pairs: expected key-val pairs: key=doc_name, val=relevance_score
    :param retrieved_pairs: retrieved key-val pairs: key=doc_name, val=relevance_score
    :param nr_of_docs: total number of docs taken into consideration
    :param relevance_threshold: min relevance score of our retrieved documents
    :return: kappa value
    """
    relevant_retrieved_pairs = {}

    # sort the retrieved pairs by relevance score
    for key in retrieved_pairs.keys():
        if retrieved_pairs[key] >= relevance_threshold:
            relevant_retrieved_pairs[key] = retrieved_pairs[key]

        # keep looping until you go under threshold
        else:
            break

    # intersection of expected and retrieved pairs
    intersection = list(set(expected_pairs.keys()).intersection(relevant_retrieved_pairs.keys()))

    # number of conflicting and non-conflicting agreements
    total_positive_agreement = len(intersection)
    ex_conflict_agreement = len(list(set(expected_pairs.keys()) - set(intersection)))
    ret_conflict_agreement = len(list(set(relevant_retrieved_pairs.keys()) - set(intersection)))
    total_negative_agreement = nr_of_docs - total_positive_agreement - ex_conflict_agreement - ret_conflict_agreement

    P0 = (total_positive_agreement + total_negative_agreement) / nr_of_docs

    # sum of each row and column
    r1 = total_positive_agreement + ex_conflict_agreement
    r2 = ret_conflict_agreement + total_negative_agreement
    c1 = total_positive_agreement + ret_conflict_agreement
    c2 = ex_conflict_agreement + total_negative_agreement
    Pe = (((r1/nr_of_docs)*(c1/nr_of_docs))+((r2/nr_of_docs)*(c2/nr_of_docs)))

    # kappa calculation (by formula)
    kappa = (P0 - Pe) / (1 - Pe)

    return kappa

def test_all_with_lucene():
    from pylucene import PyLuceneWrapper
    RELEVANT_NR_DOCS_LUCENE = 20

    doc_dict = getDocDict(filepath_video_games=variables.filepath_video_games, csv_doc_dict=variables.csv_doc_dict)
    lucene_retrieval_system = PyLuceneWrapper(documents=doc_dict)
    retrievalsystem = SPIMI(block_size_limit=10000, force_reindex=True, documents=list(doc_dict.values()), document_titles=list(doc_dict.keys()))
    # start timer
    start = time.time()
    ks = [3,5,10]
    metrics = {"precisions@": {}, "recalls@": {}, "kappas@": {}}
    for k in ks:
        metrics[f"precisions@"][k] = []
        metrics[f"recalls@"][k] = []
        metrics[f"kappas@"][k] = []

    print("Start comparing with lucene")
    # loop over all the documents
    doc_nr = 0
    document_titles = list(doc_dict.keys())
    for title, text in doc_dict.items():

        if doc_nr % 100 == 0 and doc_nr != 0:
            print(f"Document nr: {doc_nr} /", len(doc_dict))
            average_time = (time.time() - start) / (doc_nr+1)
            print(f"Average time per document: {average_time}")
            docs_left = len(doc_dict) - doc_nr
            print(f"Time left: {average_time * docs_left}")
            print(f"Average precision@3: {sum(metrics['precisions@'][3]) / len(metrics['precisions@'][3])}")
            print(f"Average kappa@3: {sum(metrics['kappas@'][3]) / len(metrics['kappas@'][3])}")
            print(f"Average kappa@5: {sum(metrics['kappas@'][5]) / len(metrics['kappas@'][5])}")
            print(f"Average kappa@10: {sum(metrics['kappas@'][10]) / len(metrics['kappas@'][10])}")
        doc_nr += 1
        start_time_ = time.time()
        # get the similar documents from the retrieval system
        similar_documents = retrievalsystem.fast_cosine_score(text.split(), k=10)
        similar_documents_titles = [(document_titles[tup[0]], tup[1]) for tup in similar_documents]

        # print(f"Retrieval system time: {time.time() - start_time_}")
        # get the similar documents from the lucene system
        # start_time_lucene = time.time()
        similar_documents_lucene = lucene_retrieval_system.search_index(text, RELEVANT_NR_DOCS_LUCENE)
        # print(f"Lucene time: {time.time() - start_time_lucene}")

        expected = [tup[0] for tup in similar_documents_lucene]
        expected_dict = dict(similar_documents_lucene)
        retrieved = [tup[0] for tup in similar_documents_titles]
        retrieved_dict = dict(similar_documents_titles)

        # calculate the metrics
        for k in ks:
            precision = calculate_average_precision(expected=expected, retrieved=retrieved, k=k)
            recall = calculate_recall(expected=expected, retrieved=retrieved, k=k)
            kappa = calculateKappa(expected_pairs=expected_dict, retrieved_pairs=retrieved_dict, nr_of_docs=k)

            metrics[f"precisions@"][k].append(precision)
            metrics[f"recalls@"][k].append(recall)
            metrics[f"kappas@"][k].append(kappa)
    print('\n')
    for k in ks:
        print(f"Mean Average precision@{k}: ", sum(metrics["precisions@"][k])/len(metrics["precisions@"][k]))
        print(f"Average recall@{k}: ", sum(metrics["recalls@"][k])/len(metrics["recalls@"][k]))

    # print("Average kappa: ", sum(metrics["kappas"]) / len(metrics["kappas"]))

    # write the metrics to a file
    with open(variables.metrics_output_file, "w") as file:
        for k in ks:
            file.write(f"Mean Average precision@{k}: {sum(metrics['precisions@'][k])/len(metrics['precisions@'][k])}\n")
            file.write(f"Average recall@{k}: {sum(metrics['recalls@'][k])/len(metrics['recalls@'][k])}\n")
        # file.write(f"Average kappa: {sum(metrics['kappas']) / len(metrics['kappas'])}\n")

def test_all():
    """
    Test the retrieval system.
    Print the average precision, recall, F1 score and kappa.
    """
    # read the ground truth labels
    ground_truth_labels: dict = read_gt(variables.filepath_path_gt)

    # read the dictionary of documents
    doc_dict = getDocDict(filepath_video_games=variables.filepath_video_games, csv_doc_dict=variables.csv_doc_dict)

    # convert the dictionary of documents to a list of documents and a list of document titles
    documents = list(doc_dict.values())
    document_titles = list(doc_dict.keys())

    # initialize the retrieval system
    retrieval_system = RelatedDocumentsRetrieval(document_titles, documents)
    retrieval_system.initialize_tf_idf_matrix(documents)

    # initialize the retrieval system with scikit-learn
    retrieval_system_scipi = RelatedDocumentsRetrieval(document_titles, documents, use_own_vectorizer=False,
                                                       use_own_cosine_similarity=False)
    retrieval_system_scipi.initialize_tf_idf_matrix(documents)

    # test the retrieval systems
    for retrievalsystem in [retrieval_system, retrieval_system_scipi]:
        # initialize the metrics
        metrics = {"precisions": [], "recalls": [], "kappas": []}
        # loop over the ground truth labels
        for title, similar_documents_gt in ground_truth_labels.items():
            # retrieve similar documents
            query_document = doc_dict[title]
            similar_documents_titles, similar_documents, scores = retrievalsystem.retrieve_similar_documents(query_document, title, 10)

            # calculate the metrics
            score_dict = dict(zip(similar_documents_titles, scores))
            par = calculate_precision(similar_documents_gt, similar_documents_titles), calculate_recall(similar_documents_gt, similar_documents_titles)
            # add the metrics to the dictionary
            metrics["precisions"].append(par[0])
            metrics["recalls"].append(par[1])

            # calculate the F1 score
            # F1 = calculateF1score(par[0], par[1])
            # add the F1 score to the dictionary
            # metrics["F1 scores"].append(F1)

            # calculate the kappa
            ones = [1] * len(similar_documents_gt)
            scored_gt = dict(zip(similar_documents_gt, ones))
            kappa = calculateKappa(scored_gt, score_dict, nr_of_docs=len(documents))
            # add the kappa to the dictionary
            metrics["kappas"].append(kappa)

        # print the metrics
        print('\n')
        print("Average precision: ", sum(metrics["precisions"])/len(metrics["precisions"]))
        print("Average recall: ", sum(metrics["recalls"])/len(metrics["recalls"]))
        print("Average F1 score: ", sum(metrics["F1 scores"])/len(metrics["F1 scores"]))
        print("Average kappa: ", sum(metrics["kappas"]) / len(metrics["kappas"]))


if __name__ == '__main__':
    # calculateIntraListSimilarity([1, 2, 3], [1, 2, 3])

    # test the retrieval system
    # test_all()

    # compare with lucene
    test_all_with_lucene()