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

    for key in retrieved_pairs.keys():
        if retrieved_pairs[key] >= relevance_threshold:
            relevant_retrieved_pairs[key] = retrieved_pairs[key]

        # keep looping until you go under threshold
        else:
            break

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

    kappa = (P0 - Pe) / (1 - Pe)

    return kappa


def testAll():
    ground_truth_labels: dict = read_gt(variables.filepath_path_gt)

    doc_dict = getDocDict(filepath_video_games=variables.filepath_video_games, csv_doc_dict=variables.csv_doc_dict)

    documents = list(doc_dict.values())
    document_titles = list(doc_dict.keys())

    retrieval_system = RelatedDocumentsRetrieval(document_titles, documents)
    retrieval_system.initialize_tf_idf_matrix(documents)

    retrieval_system_scipi = RelatedDocumentsRetrieval(document_titles, documents, use_own_vectorizer=False,
                                                       use_own_cosine_similarity=False)
    retrieval_system_scipi.initialize_tf_idf_matrix(documents)

    for retrievalsystem in [retrieval_system, retrieval_system_scipi]:
        metrics = {"precisions": [], "recalls": [], "F1 scores": [], "kappas": []}
        for title, similar_documents_gt in alive_it(ground_truth_labels.items(), title="Testing"):
            query_document = doc_dict[title]
            similar_documents_titles, similar_documents, scores = retrievalsystem.retrieve_similar_documents(query_document, title, 10)
            score_dict = dict(zip(similar_documents_titles, scores))
            par = calculatePrecisionAndRecall(similar_documents_gt, similar_documents_titles)
            metrics["precisions"].append(par[0])
            metrics["recalls"].append(par[1])
            F1 = calculateF1score(par[0], par[1])
            metrics["F1 scores"].append(F1)
            ones = [1] * len(similar_documents_gt)
            scored_gt = dict(zip(similar_documents_gt, ones))
            kappa = calculateKappa(scored_gt, score_dict, nr_of_docs=len(documents))
            metrics["kappas"].append(kappa)
        print('\n')
        print("Average precision: ", sum(metrics["precisions"])/len(metrics["precisions"]))
        print("Average recall: ", sum(metrics["recalls"])/len(metrics["recalls"]))
        print("Average F1 score: ", sum(metrics["F1 scores"])/len(metrics["F1 scores"]))
        print("Average kappa: ", sum(metrics["kappas"]) / len(metrics["kappas"]))

if __name__ == '__main__':
    # d = read_gt(variables.filepath_path_gt)  # Read ground truth
    # # print(d["Assassin's Creed IV: Black Flag"])
    #
    # # Wikipedia example: expected output is
    # # precision = 5/8 = 0.625
    # # recall = 5/12 = 0.41666...
    # exp = ['dog0', 'dog1', 'dog2', 'dog3', 'dog4', 'dog5', 'dog6', 'dog7', 'dog8', 'dog9', 'dog10', 'dog11']
    # ret = ['dog0', 'dog1', 'dog2', 'dog3', 'dog4', "cat1", "cat2", "cat3"]
    # print(calculatePrecisionAndRecall(exp, ret))
    testAll()
