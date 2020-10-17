from scipy.stats import truncnorm
from math import sqrt, log, exp, fabs, erf
import numpy as np
from matplotlib import pyplot as plt

# functions for counting covariation matrix made of consistent estimates
def s_k_l_j(m, l, j, n_k, means):
    s = 0
    for i in range(n_k):
        s += (m[l][i] - means[l]) * (m[j][i] - means[j])
    return s / (n_k - 1)

def s_k(samples, p, n_k, means):
    s_k = np.zeros((p, p))
    for y in range(p):
        for x in range(p):
            s_k[x, y] = s_k_l_j(samples, y, x, n_k, means)
    return s_k

def covariation_matrix(s1, s2, mean1, mean2):
    p = len(s1)
    n_1 = len(s1[0])
    n_2 = len(s2[0])
    s1 = s_k(s1, p, n_1, mean1)
    s2 = s_k(s2, p, n_2, mean2)
    cov = ((n_1 - 1) * s1 + (n_2 - 1) * s2) / (n_1 + n_2 - 2)
    return cov

# the main class of the task of the discriminant analysis
class BayesianClassification:
    # initialization of consistent estimates:
    # mean, covariation matrix, z1, z2, q1, q2
    # also counting mahalanobis distance and the probability of erroneous classification
    def __init__(self, sample1, sample2, mean1 = 0, mean2 = 0, cov = 0):
        self.sample1 = sample1
        self.sample2 = sample2
        self.n1 = len(sample1[0])
        self.n2 = len(sample2[0])

        if (mean1 == 0):
            self.mean1 = np.array([np.mean(row) for row in self.sample1])
            self.mean2 = np.array([np.mean(row) for row in self.sample2])
            self.cov_matrix = covariation_matrix(self.sample1, self.sample2, self.mean1, self.mean2)
        else:
            self.mean1 = mean1
            self.mean2 = mean2
            self.cov_matrix = cov

        self.a = np.linalg.inv(self.cov_matrix).dot(self.mean1 - self.mean2)
        print(self.cov_matrix)
        self.z1, self.z2 = 0, 0
        for i in range(self.n1):
            self.z1 += self.a.dot(self.sample1[:, i])
        for i in range(self.n2):
            self.z2 += self.a.dot(self.sample2[:, i])
        self.z1 /= self.n1
        self.z2 /= self.n2

        self.r1 = self.n1 / (self.n1 + self.n2)
        self.r2 = self.n2 / (self.n1 + self.n2)

        self.c = (self.z1 + self.z2) / 2 + log(self.r1 / self.r2)

        self.D = 0
        self.DH = 0
        self._culc_mahalanobis_distance()
        self.P_12 = 0
        self.P_21 = 0
        self._culc_prob()

    def _classify(self, array, is_q_set=True):
        compare = lambda arr: arr.dot(self.a) < self.c + log(self.r2 / self.r1)
        if not is_q_set:
            compare = lambda arr: arr.dot(self.a) < self.c
        if compare(array):
            return 0
        else:
            return 1

    def plot_data_3D(self, sample_1, sample_2):
        x_max = max(max(sample_1[0]), max(sample_2[0])) + 1
        x_min = min(min(sample_1[0]), min(sample_2[0])) - 1
        y_max = max(max(sample_1[1]), max(sample_2[1])) + 1
        y_min = min(min(sample_1[1]), min(sample_2[1])) - 1
        n_1 = len(sample_1[0])
        n_2 = len(sample_2[0])

        x = np.linspace(x_min, x_max, 4)
        y = np.linspace(y_min, y_max, 4)
        X, Y = np.meshgrid(x, y)
        f = lambda _x, _y: ((- self.a[0] * _x - self.a[1] * _y) + np.full(_x.shape, self.c)) / self.a[2]
        Z = f(X, Y)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        for i in range(n_1):
            x = sample_1[:, i]
            if self._classify(x) == 1:
                ax.scatter(x[0], x[1], x[2], c='teal', marker='+')
            else:
                ax.scatter(x[0], x[1], x[2], c='teal', marker='o')

        for i in range(n_2):
            x = sample_2[:, i]
            if self._classify(x) == 0:
                ax.scatter(x[0], x[1], x[2], c='mediumpurple', marker='*')
            else:
                ax.scatter(x[0], x[1], x[2], c='mediumpurple', marker='^')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.plot_wireframe(X, Y, Z, color='grey')
        plt.show()
        return

    def classify(self, sample_1, sample_2):
        q1 = 0
        q2 = 0
        n1 = len(sample_1[0])
        n2 = len(sample_2[0])

        for i in range(n1):
            x = sample_1[:, i]
            if self._classify(x) == 1:
                q1 += 1

        for i in range(n2):
            x = sample_2[:, i]
            if self._classify(x) == 0:
                q2 += 1
        print('q1 = ' + str(q1) + ' q2 = ' + str(q2))

    def _culc_prob(self):
        F = lambda x: 1 / 2 * (1 + erf(x / sqrt(2)))
        K = log(self.r2 / self.r1)

        self.P_12 = F((-K - self.DH / 2) / sqrt(self.D))

        self.P_21 = F((K - self.DH / 2) / sqrt(self.D))

    def _culc_mahalanobis_distance(self, p=3):
        _sum = (self.a.dot(self.cov_matrix)).dot(self.a)

        self.D = (self.z1 - self.z2) * (self.z1 - self.z2) / _sum
        self.DH = fabs((self.n1 + self.n2 - p - 3) / (self.n1 + self.n2 - 2) * self.D - p * (1 / self.n1 + 1 / self.n2))







