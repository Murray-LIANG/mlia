from __future__ import division

import pickle

import matplotlib.pyplot as plt

decision_node = dict(boxstyle='sawtooth', fc='0.8')
leaf_node = dict(boxstyle='round4', fc='0.8')
arrow_args = dict(arrowstyle='<-')


def plot_node(axes, node_text, from_point, to_point, node_type):
    axes.annotate(node_text, xy=from_point, xycoords='axes fraction',
                  xytext=to_point, textcoords='axes fraction', va='center',
                  ha='center', bbox=node_type, arrowprops=arrow_args)


def get_leafs_number(tree):
    root = tree.keys()[0]
    subtree = tree[root]
    res = 0
    for child in subtree:
        if isinstance(subtree[child], dict):
            res += get_leafs_number(subtree[child])
        else:
            res += 1
    return res


def get_depth(tree):
    root = tree.keys()[0]
    subtree = tree[root]
    res = 0
    for child in subtree:
        if isinstance(subtree[child], dict):
            res = max(res, 1 + get_depth(subtree[child]))
        else:
            res = 1
    return res


def store_tree(tree, file_name):
    with open(file_name, 'w') as f:
        pickle.dump(tree, f)


def grab_tree(file_name):
    with open(file_name) as f:
        return pickle.load(f)


def plot_edge(axes, edge_text, from_point, to_point):
    from_x, from_y = from_point[0], from_point[1]
    to_x, to_y = to_point[0], to_point[1]
    axes.text((from_x + to_x) / 2, (from_y + to_y) / 2, edge_text)


figure_width = figure_height = 0


def plot_tree(axes, tree, from_point, from_edge_text, offset_x, offset_y):
    root = tree.keys()[0]
    leafs_num = get_leafs_number(tree)
    to_point = (offset_x + (1 + leafs_num) / 2 / figure_width,
                offset_y)
    plot_edge(axes, from_edge_text, from_point, to_point)
    plot_node(axes, root, from_point, to_point, decision_node)
    sub_tree = tree[root]
    for child in sub_tree:
        if isinstance(sub_tree[child], dict):
            plot_tree(axes, sub_tree[child], to_point, child,
                      offset_x, offset_y - 1 / figure_height)
        else:
            offset_x += 1 / figure_width
            sub_point = (offset_x, offset_y - 1 / figure_height)
            plot_edge(axes, child, to_point, sub_point)
            plot_node(axes, sub_tree[child], to_point, sub_point, leaf_node)


def create_plot(tree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    # axis_props = dict(xticks=[], yticks=[])
    axis_props = dict()
    axes = plt.subplot(1, 1, 1, frameon=False, **axis_props)
    global figure_width, figure_height
    figure_width = get_leafs_number(tree)
    figure_height = get_depth(tree)
    # plot_node(axes, 'Decision Node', (0.5, 0.1), (0.1, 0.5), decision_node)
    # plot_node(axes, 'Leaf Node', (0.8, 0.1), (0.4, 0.8), leaf_node)
    # plot_edge(axes, (0.5, 0.1), (0.1, 0.5), 'abc')
    plot_tree(axes, tree, (0.5, 1), '', -0.5 / figure_width, 1)
    plt.show()


if __name__ == '__main__':
    create_plot(grab_tree('./tree.txt'))
