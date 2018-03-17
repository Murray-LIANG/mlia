from __future__ import division

from collections import Counter
import math
import os

import tree_plot

def calculate_entropy(data_set):
    total_num = len(data_set)
    labels_count = Counter([row[-1] for row in data_set])

    return -sum([v / total_num * math.log(v / total_num, 2)
                 for v in labels_count.values()])


def create_data_set():
    data_set = [[1, 1, 'yes'],
                [1, 1, 'yes'],
                [1, 0, 'no'],
                [0, 1, 'no'],
                [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return data_set, labels


def split_data_set(data_set, axis, value):
    return [row[:axis] + row[axis + 1:]
            for row in data_set if row[axis] == value]


def choose_best_feature(data_set):
    row_size = len(data_set)
    feature_size = len(data_set[0]) - 1  # the last one is target type
    base_entropy = calculate_entropy(data_set)

    max_info_gain = 0
    best_feature_index = -1
    for index in range(feature_size):
        feature_entropies = []
        feature_counter = Counter([row[index] for row in data_set])
        for feature_k, feature_v in feature_counter.items():
            sub_set = split_data_set(data_set, index, feature_k)
            feature_entropies.append(
                feature_v / row_size * calculate_entropy(sub_set))
        info_gain = base_entropy - sum(feature_entropies)
        if info_gain > max_info_gain:
            max_info_gain = info_gain
            best_feature_index = index

    return best_feature_index


def create_tree(data_set, labels):
    class_list = [row[-1] for row in data_set]
    if class_list.count(class_list[0]) == len(data_set):
        return class_list[0]
    if len(data_set[0]) == 1:
        return Counter(class_list).most_common(1)[0][0]

    best_feature_index = choose_best_feature(data_set)
    best_feature_label = labels[best_feature_index]
    tree = {}

    feature_counter = Counter([row[best_feature_index] for row in data_set])
    for k in feature_counter.keys():
        feature_counter[k] = create_tree(
            split_data_set(data_set, best_feature_index, k),
            labels[:best_feature_index] + labels[best_feature_index + 1:])
    tree[best_feature_label] = dict(feature_counter)
    return tree


if __name__ == '__main__':
    data_set, labels = create_data_set()
    tree = create_tree(data_set, labels)
    print(tree)
    tree_plot.store_tree(tree, './tree.txt')
