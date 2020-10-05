import numpy as np
import os
import Lab1.BayesianClassification as BC
root_path = os.path.abspath("../")
data_path = os.path.join(root_path, 'data/german.data')
if not os.path.exists(data_path):
    print(f"cannot find data file: {data_path}")
    exit(1)

p = 3
col1 = 3
col2 = 5
col3 = 8

matrix = np.loadtxt(data_path)

def get_sample(matrix, cols, n, marker, p, start_from = 0):
    sample = np.zeros((p, n))
    col1_i = 0
    offset = 0
    for i in range(1000):
        if col1_i >= n:
            break
        if col1_i + offset < start_from:
            offset += 1
            continue
        if matrix[i][24] == marker and col1_i < n:
            for j in range(p):
                sample[j][col1_i] = matrix[i][cols[j]]
            col1_i += 1
    return sample

n1 = 100
sample1 = get_sample(matrix, [col1, col2, col3], n1, marker=1, p=p)
n2 = 100
sample2 = get_sample(matrix, [col1, col2, col3], n2, marker=2, p=p)
BC = BC.BayesianClassification(sample1, sample2)
BC.plot_data(sample1, sample2)

n3 = 10
sample3 = get_sample(matrix, [col1, col2, col3], n3, marker=1, p=p, start_from=20)
n4 = 10
sample4 = get_sample(matrix, [col1, col2, col3], n4, marker=2, p=p, start_from=20)
BC.plot_data(sample3, sample4)