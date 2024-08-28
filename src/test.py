"""
This file contains the evaluation functions for the retrieval system.
"""

import copy
import numpy as np
from src.inverted_index import SPIMI
from utils import *
import time
import variables


def calculate_confusion_matrix(expected: list, retrieved: list) -> tuple:
    """
    Calculate the confusion matrix.

    Args:
        expected (list): list of expected values (ground truth labels)
        retrieved (list): list of retrieved values (our algorithm)

    Returns:
        tuple: (TP, FP, FN)
    """
    # number of retrieved values that are also in expected (True positive)
    TP = len([ret for ret in retrieved if ret in expected])
    # number of retrieved values that aren't in expected (False positive)
    FP = len([ret for ret in retrieved if ret not in expected])
    # number of expected values that aren't retrieved (False negative)
    FN = len([ex for ex in expected if ex not in retrieved])

    return TP, FP, FN

def calculate_precision(expected, retrieved, k=0) -> float:
    """
    • P(recision) = TP/(TP+FP) how much correct of found

    Args:
        expected (list): list of expected values (ground truth labels)
        retrieved (list): list of retrieved values (our algorithm)
        k (int): cut-off value (top k retrieved values)

    Returns:
        float: precision
    """
    if k != 0:
        retrieved = retrieved[:k]
    TP, FP, FN = calculate_confusion_matrix(expected, retrieved)

    # precision calculation (by formula)
    precision = TP / (TP + FP) if TP + FP != 0 else 0

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
    recall = TP / (TP + FN) if TP + FN != 0 else 0

    return recall


def calculate_average_precision(expected, retrieved, k=0):
    if k == 0:
        k = len(retrieved)
    average_precision_at_k = 0
    for i in range(0, k):
        if retrieved[i] in expected:
            precision_at_k = calculate_precision(expected, retrieved, k=i + 1)
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


def calculateKappa(expected_pairs: dict, retrieved_pairs: dict, nr_of_docs, relevance_threshold=0.85):
    """
    kappa >= 2/3 is good
    :param expected_pairs: expected key-val pairs: key=doc_name, val=relevance_score
    :param retrieved_pairs: retrieved key-val pairs: key=doc_name, val=relevance_score
    :param nr_of_docs: total number of docs taken into consideration
    :param relevance_threshold: min relevance score of our retrieved documents
    :return: kappa value
    """
    relevant_retrieved_pairs = {}
    old = copy.deepcopy(expected_pairs)
    expected_pairs = dict()
    j = 0
    for k in old.keys():
        j += 1
        expected_pairs[k] = old[k]
        if j > nr_of_docs:
            break
    nr_of_docs = len(set(list(expected_pairs.keys()) + list(retrieved_pairs.keys())))

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
    Pe = (((r1 / nr_of_docs) * (c1 / nr_of_docs)) + ((r2 / nr_of_docs) * (c2 / nr_of_docs)))

    # kappa calculation (by formula)
    if 1 - Pe == 0:
        kappa = 1
    else:
        kappa = (P0 - Pe) / (1 - Pe)

    # if kappa > 1:
    #     # print(expected_pairs)
    #     #print(Pe)
    #     if Pe == 2.0:
    #         b = 2
    #     a = 2
    # print(kappa)

    return kappa


def initialize_metrics(ks):
    metrics = {"precisions@": {}, "recalls@": {}, "kappas@": {}}
    for k in ks:
        metrics[f"precisions@"][k] = []
        metrics[f"recalls@"][k] = []
        metrics[f"kappas@"][k] = []

    return metrics


def test_all_with_lucene():
    from pylucene import PyLuceneWrapper
    all_metrics = []
    ks = [3, 5, 10]
    LUCENE_RELEVANT_NR_DOCS_LIST = [10, 20]
    doc_dict = getDocDict(filepath_video_games=variables.filepath_video_games, csv_doc_dict=variables.csv_doc_dict)
    lucene_retrieval_system = PyLuceneWrapper(documents=doc_dict)
    retrievalsystem = SPIMI(block_size_limit=10000, force_reindex=True, documents=list(doc_dict.values()),
                            document_titles=list(doc_dict.keys()))

    for RELEVANT_NR_DOCS_LUCENE in LUCENE_RELEVANT_NR_DOCS_LIST:

        # start timer
        start = time.time()
        metrics = initialize_metrics(ks)

        print("Start comparing with lucene")
        # loop over all the documents
        doc_nr = 0
        document_titles = list(doc_dict.keys())
        for title, text in doc_dict.items():

            if doc_nr % 100 == 0 and doc_nr != 0:
                print(f"Document nr: {doc_nr} /", len(doc_dict))
                average_time = (time.time() - start) / (doc_nr + 1)
                print(f"Average time per document: {average_time}")
                docs_left = len(doc_dict) - doc_nr
                print(f"Time left: {average_time * docs_left}")
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
            print(f"Mean Average precision@{k}: ", sum(metrics["precisions@"][k]) / len(metrics["precisions@"][k]))
            print(f"Average recall@{k}: ", sum(metrics["recalls@"][k]) / len(metrics["recalls@"][k]))
            print(f"Average kappa@{k}: ", sum(metrics["kappas@"][k]) / len(metrics["kappas@"][k]))

        all_metrics.append(metrics)

    # print("Average kappa: ", sum(metrics["kappas"]) / len(metrics["kappas"]))
    # write the metrics to a file
    with open(variables.metrics_output_file, "w") as file:
        for metric in all_metrics:
            for k in ks:
                file.write(
                    f"Mean Average precision@{k}: {sum(metric['precisions@'][k]) / len(metric['precisions@'][k])}\n")
                file.write(f"Average recall@{k}: {sum(metric['recalls@'][k]) / len(metric['recalls@'][k])}\n")
                file.write(f"Average kappa@{k}: {sum(metric['kappas@'][k]) / len(metric['kappas@'][k])}\n")
            file.write("\n----------\n")

def plot_metrics(metrics, ks):
    import matplotlib.pyplot as plt
    system_retrieval_names = metrics.keys()

    for k in ks:
        for group in system_retrieval_names:

            plt.plot(metrics[group]["precisions@"][k], label=f"Precision@{k}", color='r')
            plt.plot(metrics[group]["recalls@"][k], label=f"Recall@{k}", color='b')
            plt.plot(metrics[group]["kappas@"][k], label=f"Kappa@{k}", color='g')
            # draw the averages
            plt.axhline(y=metrics[group][f"Mean Average precision@{k}"], color='r', linestyle='--',
                        label=f"Mean Average precision")
            plt.axhline(y=metrics[group][f"Average recall@{k}"], color='b', linestyle='--', label=f"Average recall")
            plt.axhline(y=metrics[group][f"Average kappa@{k}"], color='g', linestyle='--', label=f"Average kappa")

            plt.legend()
            plt.title(f"{group} metrics for k={k}")
            plt.savefig(f"{variables.metric_plot_path}metrics_{group}_{k}.png")
            # close the plot
            plt.close()
    system_retrieval_names = [group['retrieval_system_name'] for group in metrics.values()]
    for k in ks:
        x = np.arange(len(system_retrieval_names))  # the label locations
        width = 0.25  # the width of the bars
        multiplier = 0
        fig, ax = plt.subplots(layout='constrained')
        entries = {
            f"Mean Average precision@{k}": [],
            f"Average recall@{k}": [],
            f"Average kappa@{k}": []
        }
        for group in metrics.keys():
            # Plot grouped bar charts of the averages of the metrics, grouped by system
            entries[f"Mean Average precision@{k}"].append(metrics[group][f"Mean Average precision@{k}"])
            entries[f"Average recall@{k}"].append(metrics[group][f"Average recall@{k}"])
            entries[f"Average kappa@{k}"].append(metrics[group][f"Average kappa@{k}"])
        colors = ['blue', 'orange', 'grey']
        for attribute, measurement in entries.items():
            offset = width * multiplier
            rects = ax.bar(x + offset, measurement, width, label=attribute, color=colors[multiplier])
            # only print two decimals
            ax.bar_label(rects, padding=3, fontsize=12, fmt='%.2f')

            multiplier += 1
        # horizontal gridlines
        ax.yaxis.grid(True)
        plt.title(f"Average metrics against ground truths", fontsize=16)
        # make all fonts bigger
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        # y-axis label fontsize
        plt.ylabel('Score', fontsize=14)





        ax.set_xticks(x + width, system_retrieval_names)
        ax.legend(fontsize=12)
        plt.savefig(f"{variables.metric_plot_path}metrics_{k}.png")


def test_all():
    """
    Test the retrieval system.
    Print the average precision, recall, F1 score and kappa.
    """
    from pylucene import PyLuceneWrapper
    ks = [3, 5, 10]

    # read the ground truth labels
    ground_truth_labels: dict = read_gt(variables.filepath_path_gt)

    # initialize the retrieval systems
    doc_dict = getDocDict(filepath_video_games=variables.filepath_video_games, csv_doc_dict=variables.csv_doc_dict)
    lucene_retrieval_system = PyLuceneWrapper(documents=doc_dict)
    retrievalsystem = SPIMI(block_size_limit=10000, force_reindex=True, documents=list(doc_dict.values()),
                            document_titles=list(doc_dict.keys()))
    print("Start comparing with ground truth")
    metrics = {"lucene": initialize_metrics(ks), "spimi": initialize_metrics(ks)}
    metrics["spimi"]["retrieval_system_name"] = "SPIMI"
    metrics["lucene"]["retrieval_system_name"] = "Lucene"

    # loop over the ground truth labels
    doc_nr = 0
    document_titles = list(doc_dict.keys())
    start = time.time()

    for title, similar_documents_gt in ground_truth_labels.items():
        if doc_nr % 10 == 0 and doc_nr != 0:
            print(f"Document nr: {doc_nr} /", len(ground_truth_labels))
            average_time = (time.time() - start) / (doc_nr + 1)
            print(f"Average time per document: {average_time}")
            docs_left = len(ground_truth_labels) - doc_nr
            print(f"Time left: {average_time * docs_left}")
        doc_nr += 1
        start_time_ = time.time()

        # get the text of the document
        text = doc_dict[title]

        # get the similar documents from the retrieval system
        similar_documents = retrievalsystem.fast_cosine_score(text.split(), k=10)
        similar_documents_titles = [(document_titles[tup[0]], tup[1]) for tup in similar_documents]

        # print(f"Retrieval system time: {time.time() - start_time_}")
        # get the similar documents from the lucene system
        # start_time_lucene = time.time()
        similar_documents_lucene = lucene_retrieval_system.search_index(text, 10)
        # print(f"Lucene time: {time.time() - start_time_lucene}")



        retrieved_lucene = [tup[0] for tup in similar_documents_lucene]
        retrieved_lucene_dict = dict(similar_documents_lucene)
        retrieved = [tup[0] for tup in similar_documents_titles]
        retrieved_dict = dict(similar_documents_titles)
        similar_documents_gt_dict = dict(similar_documents_gt)

        # calculate the metrics
        for k in ks:
            precision_rerieval = calculate_average_precision(expected=similar_documents_gt, retrieved=retrieved, k=k)
            recall_rerieval = calculate_recall(expected=similar_documents_gt, retrieved=retrieved, k=k)
            kappa_rerieval = calculateKappa(expected_pairs=similar_documents_gt_dict, retrieved_pairs=retrieved_dict, nr_of_docs=k)

            precision_lucene = calculate_average_precision(expected=similar_documents_gt, retrieved=retrieved_lucene, k=k)
            recall_lucene = calculate_recall(expected=similar_documents_gt, retrieved=retrieved_lucene, k=k)
            kappa_lucene = calculateKappa(expected_pairs=similar_documents_gt_dict, retrieved_pairs=retrieved_lucene_dict, nr_of_docs=k)


            metrics["spimi"][f"precisions@"][k].append(precision_rerieval)
            metrics["spimi"][f"recalls@"][k].append(recall_rerieval)
            metrics["spimi"][f"kappas@"][k].append(kappa_rerieval)

            metrics["lucene"][f"precisions@"][k].append(precision_lucene)
            metrics["lucene"][f"recalls@"][k].append(recall_lucene)
            metrics["lucene"][f"kappas@"][k].append(kappa_lucene)
    print('\n')
    for retrieval in ["spimi", "lucene"]:
        for k in ks:
            metrics[retrieval][f"Mean Average precision@{k}"] = sum(metrics[retrieval]["precisions@"][k]) / len(metrics[retrieval]["precisions@"][k])
            metrics[retrieval][f"Average recall@{k}"] = sum(metrics[retrieval]["recalls@"][k]) / len(metrics[retrieval]["recalls@"][k])
            metrics[retrieval][f"Average kappa@{k}"] = sum(metrics[retrieval]["kappas@"][k]) / len(metrics[retrieval]["kappas@"][k])
            print(f"Mean Average precision@{k}: ", metrics[retrieval][f"Mean Average precision@{k}"])
            print(f"Average recall@{k}: ", metrics[retrieval][f"Average recall@{k}"])
            print(f"Average kappa@{k}: ", metrics[retrieval][f"Average kappa@{k}"])
    plot_metrics(metrics, ks)

if __name__ == '__main__':

    ### test the retrieval system ###

    # compare with ground truths
    # test_all()

    # compare with lucene
    test_all()
