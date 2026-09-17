"""
Exercise 3.16
Based on the example program from Example 3.22, design your own
eight-state routing diagram, and modify the program from Example 3.22
accordingly to solve your routing problem.
"""
import numpy as np
import matplotlib.pyplot as plt
from Q_Utils import *

# Custom 8-state (0-7) routing diagram; goal = state 7
points_list = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4), (4, 5), (4, 6), (5, 7), (6, 7)]
goal = 7

# Show the routing graph
showgraph(points_list)

MATRIX_SIZE = 8
R = createRmat(MATRIX_SIZE, points_list, goal)
Q = np.matrix(np.zeros([MATRIX_SIZE, MATRIX_SIZE]))
gamma = 0.8

scores = []
for i in range(700):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(R, current_state)
    action = sample_next_action(available_act)
    score = update(R, Q, current_state, action, gamma)
    scores.append(score)
    print('Score:', str(score))

print("Trained Q matrix:")
print(Q / np.max(Q) * 100)

current_state = 0
steps = [current_state]
while current_state != goal:
    next_step_index = np.where(Q[current_state, ] == np.max(Q[current_state, ]))[1]
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1))
    else:
        next_step_index = int(next_step_index)
    steps.append(next_step_index)
    current_state = next_step_index

print("Most efficient path:")
print(steps)

plt.plot(scores)
plt.show()


"""
Output aktual (hasil eksekusi nyata):
Score: 0
Score: 0
Score: 0
Score: 0
Score: 0
Score: 0
Score: 0
Score: 0
Score: 0
Score: 0
... (dipotong agar ringkas) ...
 [  0.           0.           0.           0.          63.98130847
    0.           0.          99.97079448]
 [  0.           0.           0.           0.           0.
   79.94743007  79.94743007 100.        ]]
Exercise_3_16.py:42: DeprecationWarning: Conversion of an array with ndim > 0 to a scalar is deprecated, and will error in future. Ensure you extract a single element from your array before performing this operation. (Deprecated NumPy 1.25.)
  next_step_index = int(next_step_index)
Most efficient path:
[0, 2, 3, 4, 6, 7]
"""
