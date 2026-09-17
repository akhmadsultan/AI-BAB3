"""
Q_Utils.py - helper functions for Example 3.22 (Q-learning routing demo).

This module is referenced by Example 3.22 in the book but its source is not
printed there (the book says it is "modified from" a few online Q-learning
tutorials). This is a standard reimplementation of the usual helper
functions (graph plotting, R-matrix construction, and the classic Q-learning
update rule) so the example can actually be run and its output observed.
"""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def showgraph(points_list):
    G = nx.Graph()
    G.add_edges_from(points_list)
    pos = nx.spring_layout(G, seed=42)
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_edges(G, pos)
    nx.draw_networkx_labels(G, pos)
    plt.title("Routing graph")
    plt.show()


def createRmat(MATRIX_SIZE, points_list, goal):
    R = np.full((MATRIX_SIZE, MATRIX_SIZE), -1.0)
    for point in points_list:
        i, j = point
        if j == goal:
            R[i, j] = 100.0
        else:
            R[i, j] = 0.0
        if i == goal:
            R[j, i] = 100.0
        else:
            R[j, i] = 0.0
    R[goal, goal] = 100.0
    return R


def available_actions(R, state):
    current_row = R[state, ]
    return np.where(current_row >= 0)[1] if current_row.ndim > 1 else np.where(current_row >= 0)[0]


def sample_next_action(available_actions_range):
    return int(np.random.choice(available_actions_range, 1)[0])


def update(R, Q, current_state, action, gamma):
    max_index = np.where(Q[action, ] == np.max(Q[action, ]))[1] if Q[action, ].ndim > 1 else np.where(Q[action, ] == np.max(Q[action, ]))[0]
    max_index = int(np.random.choice(max_index, size=1)[0]) if max_index.size > 1 else int(max_index[0])
    max_value = Q[action, max_index]
    Q[current_state, action] = R[current_state, action] + gamma * max_value
    if np.max(Q) > 0:
        return np.sum(Q / np.max(Q) * 100)
    return 0
