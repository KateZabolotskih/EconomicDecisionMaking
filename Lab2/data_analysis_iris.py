import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from math import sqrt
from Lab2.principal_component_method import PCA

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

iris = datasets.load_iris()
data = np.array(iris.data[:, :]).transpose()
clazz = iris.target

standardized_data = normalized(data)
print(np.cov(standardized_data))
print(np.mean(standardized_data))
pca = PCA(standardized_data)
pca_data = pca.Z()
print(np.cov(pca_data))

points = pca_data.transpose()

colors = ['g', 'c', 'y']
markers = ['X', '.', '*']
for i in range(points.shape[0]):
    x = points[i]
    c = clazz[i]
    plt.plot(x[0], x[1], f'{colors[c]}{markers[c]}')
plt.show()

print(get_percentages_of_significance(np.cov(pca_data)))