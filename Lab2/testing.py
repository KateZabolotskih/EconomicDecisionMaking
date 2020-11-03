import numpy as np
import os
from Lab2.principal_component_method import PCA
from matplotlib import pyplot as plt
from math import sqrt, fabs

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

def run_PCA_norm(dim, mean, covmat):
    sample = np.random.multivariate_normal(mean, covmat, 100).transpose()
    print("covmat X", dim, "D № 1")
    print(np.cov(normalized(sample)))
    print(' ')
    pca = PCA(normalized(sample))
    z = pca.Z()
    print("covmat Z", dim, "D № 1")
    print(np.cov(z))
    print(' ')

def plot_2D(sample: '[[], []]'):
    plt.scatter(sample[0], sample[1], c="gold")
    pca = PCA(sample)
    k1 = sqrt(fabs(pca.eigenvalue[0]))
    k2 = sqrt(fabs(pca.eigenvalue[1]))
    t = np.linspace(0, 1, 100)
    x_ev = t * pca.eigenvectors[0][0] * k1
    y_ev = t * pca.eigenvectors[1][0] * k1
    plt.plot(x_ev, y_ev, linewidth=2, c='tomato')
    x_ev = t * pca.eigenvectors[0][1] * k2
    y_ev = t * pca.eigenvectors[1][1] * k2
    plt.plot(x_ev, y_ev, linewidth=2, c='tomato', )
    plt.show()

def plot_3D(sample: '[[], [], []]'):
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(sample[0], sample[1], sample[2], c="mediumpurple", marker='.')
    pca = PCA(sample)
    k1 = sqrt(fabs(pca.eigenvalue[0]))
    k2 = sqrt(fabs(pca.eigenvalue[1]))
    k3 = sqrt(fabs(pca.eigenvalue[2]))
    t = np.linspace(0, 1, 100)
    x_ev = t * pca.eigenvectors[0][0] * k1
    y_ev = t * pca.eigenvectors[1][0] * k1
    z_ev = t * pca.eigenvectors[2][0] * k1
    ax.plot(x_ev, y_ev, z_ev, linewidth=3, c='teal')
    x_ev = t * pca.eigenvectors[0][1] * k2
    y_ev = t * pca.eigenvectors[1][1] * k2
    z_ev = t * pca.eigenvectors[2][1] * k2
    ax.plot(x_ev, y_ev, z_ev, linewidth=3, c='teal')
    x_ev = t * pca.eigenvectors[0][2] * k3
    y_ev = t * pca.eigenvectors[1][2] * k3
    z_ev = t * pca.eigenvectors[2][2] * k3
    ax.plot(x_ev, y_ev, z_ev, linewidth=3,  c='teal')
    plt.show()

#twodimensional
mean = (0, 0)

covmat_1 = [[10, 10],
            [10, 10]]

covmat_2 = [[10, 2],
            [2, 10]]
print("good correlation")
run_PCA_norm(2, mean, covmat_1)
print("bad correlation")
run_PCA_norm(2, mean, covmat_2)

sample_1 = np.random.multivariate_normal(mean, covmat_1, 100).transpose()
plot_2D(normalized(sample_1))
sample_2 = np.random.multivariate_normal(mean, covmat_2, 100).transpose()
plot_2D(normalized(sample_2))
#threedimensional
mean = (0, 0, 0)

covmat_1 = [[22, 22,  1],
            [22, 22,  1],
            [ 1,  1, 22]]

covmat_2 = [[22,  3,  3],
            [ 3, 22,  3],
            [ 3,  3, 22]]

print("good correlation")
run_PCA_norm(3, mean, covmat_1)
sample_1 = np.random.multivariate_normal(mean, covmat_1, 100).transpose()
plot_3D(normalized(sample_1))
print("bad correlation")
run_PCA_norm(3, mean, covmat_2)
sample_2 = np.random.multivariate_normal(mean, covmat_2, 100).transpose()
plot_3D(normalized(sample_2))