import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np


def create_data_set():
    data_set = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return data_set, labels


data_set, labels = create_data_set()


def classify(input, data_set, labels, k):
    """

    :param input:
    :param data_set:
    :param labels:
    :param k:
    :return:
    """
    data_rows = data_set.shape[0]
    input = np.tile(input, (data_rows, 1))
    sq_diff = (input - data_set) ** 2
    distances = sq_diff.sum(axis=1) ** 0.5
    sorted_indices = distances.argsort()
    sorted_labels = [labels[sorted_indices[i]] for i in range(k)]
    return Counter(sorted_labels).most_common(1)[0][0]


def read_from_file(file_name):
    with open(file_name) as f:
        lines = f.readlines()

    matrix = np.zeros((len(lines), 3))
    labels = []
    for index, line in enumerate(lines):
        parts = line.strip().split('\t')
        matrix[index, :] = parts[:3]
        labels.append(int(parts[-1]))
    return matrix, labels


DATA_ROOT_DIR = './MLiA_SourceCode/machinelearninginaction/'
data_set, labels = read_from_file(
    os.path.join(DATA_ROOT_DIR, 'Ch02/datingTestSet2.txt'))

# print(data_set, labels)
# print(labels)
# print(np.array(labels).dtype)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(data_set[:, 0], data_set[:, 1],
           s=10 * np.array(labels),
           c=15 * np.array(labels))


# plt.show()


def normalize(data_set):
    min_vals = data_set.min(0)
    max_vals = data_set.max(0)
    ranges = max_vals - min_vals
    rows = data_set.shape[0]

    normalized = ((data_set - np.tile(min_vals, (rows, 1))) /
                  np.tile(ranges, (rows, 1)))

    return normalized, ranges, min_vals


def dating_class_test():
    test_data_ratio = 0.1
    dating_matrix, dating_labels = read_from_file(
        os.path.join(DATA_ROOT_DIR, 'Ch02/datingTestSet2.txt'))
    norm_matrix, ranges, min_vals = normalize(dating_matrix)
    rows_all = dating_matrix.shape[0]
    rows_test = int(test_data_ratio * rows_all)

    mismatch_count = 0
    for i in range(rows_test):
        dating_class = classify(dating_matrix[i, :],
                                norm_matrix[rows_test:, :],
                                dating_labels[rows_test:],
                                3)
        print('The class calculated: {calc}, expected: {expe}.'
              .format(calc=dating_class, expe=dating_labels[i]))
        if dating_class != dating_labels[i]:
            mismatch_count += 1

    print('The total error rate: {:f}.'
          .format(mismatch_count / float(rows_test)))


# dating_class_test()


def image_to_vector(file_name):
    res = np.zeros((1, 32 ** 2))
    with open(file_name) as f:
        lines = f.readlines()

    for i in range(32):
        for j in range(32):
            res[0, 32 * i + j] = int(lines[i][j])
    return res


DIGIT_ROOT_DIR = os.path.join(DATA_ROOT_DIR, 'Ch02')
TRAINING_DIGIT_DIR = os.path.join(DIGIT_ROOT_DIR, 'trainingDigits')


def get_label(file_name):
    return file_name.split('.')[0].split('_')[0]


def handwriting_class_test():
    training_files = os.listdir(TRAINING_DIGIT_DIR)
    training_rows = len(training_files)
    training_matrix = np.zeros((training_rows, 1024))
    training_labels = []

    for index, training_file in enumerate(training_files):
        training_labels.append(int(get_label(training_file)))
        training_matrix[index, :] = image_to_vector(
            os.path.join(TRAINING_DIGIT_DIR, training_file))

    test_files = os.listdir(TRAINING_DIGIT_DIR)
    test_rows = len(test_files)
    mismatch_count = 0
    for index, test_file in enumerate(test_files):
        test_label = int(get_label(test_file))
        test_vector = image_to_vector(os.path.join(TRAINING_DIGIT_DIR,
                                                   test_file))
        digit_class = classify(test_vector, training_matrix, training_labels,
                               3)
        print('The class calculated: {calc}, expected: {expe}.'
              .format(calc=digit_class, expe=test_label))
        if digit_class != test_label:
            mismatch_count += 1

    print('The total error rate: {:f}.'
          .format(mismatch_count / float(test_rows)))


handwriting_class_test()
