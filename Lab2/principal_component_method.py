import numpy as np
class PCA:
    def __init__(self, x: '[[],[], ...]'):
        self.x = x
        self.Xcentered = self._center(x)
        self.m = len(x)
        self.covmat = np.cov(x)
        self.eigenvalue, self.eigenvectors = np.linalg.eig(self.covmat)
        self.eigenvectors = -self.eigenvectors[:,1]

    def _center(self, x: '[[],[], ...]') -> '[[], [], ...]':
        Xcentered = []
        for i in range(self.m):
            mean = np.mean(x[i])
            Xcentered.append(x[i] - mean)
        return Xcentered

    def Z(self) -> '[[],[], ...]':
        return self.eigenvectors.transpose().dot(self.Xcentered)



