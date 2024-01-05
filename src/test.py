from src.related_doc_retrieval import RelatedDocumentsRetrieval
from utils import *
import variables


def calculatePrecisionAndRecall(expected, retrieved) -> tuple:
    """
    • P(recision) = TP/(TP+FP) how much correct of found
    • R(ecall) = TP/(TP+FN) how much of correct
    :param expected: list of expected values (ground truth labels)
    :param retrieved: list of retrieved values (our algorithm)
    :return: tuple(precision, recall)
    """
    # number of retrieved values that are also in expected (True positive)
    TP = len([ret for ret in retrieved if ret in expected])

    # number of retrieved values that aren't in expected (False positive)
    FP = len([ret for ret in retrieved if ret not in expected])

    # number of expected values that aren't retrieved (False negative)
    FN = len([ex for ex in expected if ex not in retrieved])

    # precision and recall calculation (by formula)
    precision = TP/(TP+FP) if TP+FP != 0 else 0
    recall = TP/(TP+FN) if TP+FN != 0 else 0

    return precision, recall


def calculateF1score(precision, recall) -> float:
    """
    F1 score calculation (by formula)
    F1 = 2*P*R/(P+R)
    :param precision: P
    :param recall: R
    :return: F1 score
    """
    # F1 score calculation (by formula)
    F1_score = (2*precision*recall) + (precision+recall)
    return F1_score


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


def testAll():
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
        metrics = {"precisions": [], "recalls": [], "F1 scores": [], "kappas": []}
        # loop over the ground truth labels
        for title, similar_documents_gt in alive_it(ground_truth_labels.items(), title="Testing"):
            # retrieve similar documents
            query_document = doc_dict[title]
            similar_documents_titles, similar_documents, scores = retrievalsystem.retrieve_similar_documents(query_document, title, 10)
            # calculate the metrics
            score_dict = dict(zip(similar_documents_titles, scores))
            par = calculatePrecisionAndRecall(similar_documents_gt, similar_documents_titles)
            # add the metrics to the dictionary
            metrics["precisions"].append(par[0])
            metrics["recalls"].append(par[1])
            # calculate the F1 score
            F1 = calculateF1score(par[0], par[1])
            # add the F1 score to the dictionary
            metrics["F1 scores"].append(F1)
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
    # test the retrieval system
    testAll()
