import numpy as np
import cvxopt



class SVM:

    # exp(-||x1-x2||**2 / 2/ gamma)

    def set_kernel(self, kernel):
        self.kernel = kernel

    def fit(self, X, y, C=100000):
        n_samples, n_features = X.shape
        # P = X^T X
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                # if self.kernel is not None:
                #    K[i, j] = self.kernel(X[i], X[j])
                K[i, j] = np.dot(X[i], X[j])
        P = cvxopt.matrix(np.outer(y, y) * K)
        # q = -1 (1xN)
        q = cvxopt.matrix(np.ones(n_samples) * -1)
        # A = y^T
        A = cvxopt.matrix(y, (1, n_samples))
        # b = 0
        b = cvxopt.matrix(0.0)
        # -1 (NxN)
        G = cvxopt.matrix(
            np.concatenate(
                (np.diag(np.ones(n_samples) * -1), np.diag(np.ones(n_samples) * 1)),
                axis=0
            )
        )
        # 0 (1xN)
        h = cvxopt.matrix(
            np.concatenate((np.zeros(n_samples), np.full(n_samples, C)), axis=0)
        )
        self.solution = solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        # Lagrange multipliers
        a = np.ravel(solution['x'])
        # Lagrange have non zero lagrange multipliers
        sv = a > 1e-3
        ind = np.arange(len(a))[sv]
        self.a = a[sv]
        self.sv = X[sv]
        self.sv_y = y[sv]
        # Intercept
        self.b = 0
        for n in range(len(self.a)):
            self.b += self.sv_y[n]
            self.b -= np.sum(self.a * self.sv_y * K[ind[n], sv])
        self.b /= len(self.a)
        # Weights
        self.w = np.zeros(n_features)
        for n in range(len(self.a)):
            self.w += self.a[n] * self.sv_y[n] * self.sv[n]

    def project(self, X):
        return np.dot(X, self.w) + self.b

    def predict(self, X):
        return np.sign(self.project(X))


