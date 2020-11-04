import numpy as np
import os
from math import sqrt
from Lab2.principal_component_method import PCA
root_path = os.path.abspath("../")
data_path = os.path.join(root_path, 'data/german.data')
if not os.path.exists(data_path):
    print(f"cannot find data file: {data_path}")
    exit(1)

def normalized(sample):
    signs = len(sample)
    sample_len = len(sample[0])

    vars = [sqrt(np.var(sign_sample)) for sign_sample in sample]

    normalized_sample = list(list())
    for sing in range(signs):
        normalized_sample.append(list())
        for num in range(sample_len):
            normalized_sample[sing].append(sample[sing][num] / vars[sing])
    return normalized_sample

def get_percentages_of_significance(covmat: '[[], [], ...]') ->'[]':
    col = len(covmat)
    sum = 0
    res = []
    for i in range(col):
        sum += covmat[i][i]
    for i in range(col):
        res.append(covmat[i][i]/sum * 100)
    return res

def get_sample_from_data(matrix, cols, n, marker, p, start_from = 0):
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

m = 23
signs = []
for i in range(m):
    signs.append(i)
n = 100
matrix = np.loadtxt(data_path)
sample = get_sample_from_data(matrix, signs, n, marker=2, p=m)
pca = PCA(normalized(sample))
Z = pca.Z()
percents = get_percentages_of_significance(np.cov(Z))
print(len(percents))
print(percents)