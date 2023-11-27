from utils import *

filepath_path_gt = 'input/gt'

def calculatePrecisionAndRecall(expected, retrieved) -> tuple:
    """
    • P(recision) = TP/(TP+FP) how much correct of found
    • R(ecall) = TP/(TP+FN) how much of correct
    :param expected: list of expected values (ground truth labels)
    :param retrieved: list of retrieved values (our algorithm)
    :return: tuple(precision, retrieved)
    """
    # number of retrieved values that are also in expected (True positive)
    TP = len([ret for ret in retrieved if ret in expected])

    # number of retrieved values that aren't in expected (False positive)
    FP = len([ret for ret in retrieved if ret not in expected])

    # number of expected values that aren't retrieved (False negative)
    FN = len([ex for ex in expected if ex not in retrieved])

    # precision and recall calculation (by formula)
    precision = TP/(TP+FP)
    recall = TP/(TP+FN)

    return precision, recall


if __name__ == '__main__':
    d = read_gt(filepath_path_gt)  # Read ground truth
    # print(d["Assassin's Creed IV: Black Flag"])