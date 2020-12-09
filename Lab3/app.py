from Lab3.svm import SVM
import numpy as np
from math import fabs
from sklearn.datasets.samples_generator import make_blobs
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix



class Lab3:
    @staticmethod
    def part1():
        mean1 = (-1, 1)
        mean2 = (2, -2)
        cov = [[1, 0.7],
               [0.7, 1]]
        size1, size2 = 100, 100
        sample1 = np.random.multivariate_normal(mean1, cov, size1).transpose()
        sample2 = np.random.multivariate_normal(mean2, cov, size2).transpose()
        sample = np.concatenate((sample1, sample2), axis=1)
        clazzes = np.zeros(size1 + size2)
        for i in range(size1):
            clazzes[i] = 1
        for i in range(size1, size1 + size2):
            clazzes[i] = -1
        svm = SVM()
        svm.fit(sample.transpose(), clazzes)

        plt.figure(2)

        def f(x, w, b, c=0):
            return (-w[0] * x - b + c) / w[1]

        # w.x + b = 0
        a0 = -4
        a1 = f(a0, svm.w, svm.b)
        b0 = 4
        b1 = f(b0, svm.w, svm.b)
        plt.plot([a0, b0], [a1, b1], 'k')
        # w.x + b = 1
        a0 = -4
        a1 = f(a0, svm.w, svm.b, 1)
        b0 = 4
        b1 = f(b0, svm.w, svm.b, 1)
        plt.plot([a0, b0], [a1, b1], 'k--')
        # w.x + b = -1
        a0 = -4
        a1 = f(a0, svm.w, svm.b, -1)
        b0 = 4
        b1 = f(b0, svm.w, svm.b, -1)
        plt.plot([a0, b0], [a1, b1], 'k--')

        plt.scatter(sample.transpose()[:, 0], sample.transpose()[:, 1], c=clazzes, marker='.', cmap='tab20',
                    label="эл-ты выборки")
        plt.scatter(svm.sv[:, 0], svm.sv[:, 1], c=svm.sv_y, marker='*', cmap='tab20',
                    label="соб. в-ры")

        plt.legend()
        plt.grid(True)
        plt.show()

        print(svm.w)
        print(2 / np.linalg.norm(svm.w))

    @staticmethod
    def part2():
        def f(x, w, b, c=0):
            return (-w[0] * x - b + c) / w[1]

        def plot():
            a0 = -4
            a1 = f(a0, svm.w, svm.b)
            b0 = 4
            b1 = f(b0, svm.w, svm.b)
            ax.plot([a0, b0], [a1, b1], 'k')
            a0 = -4
            a1 = f(a0, svm.w, svm.b, 1)
            b0 = 4
            b1 = f(b0, svm.w, svm.b, 1)
            ax.plot([a0, b0], [a1, b1], 'k--')
            a0 = -4
            a1 = f(a0, svm.w, svm.b, -1)
            b0 = 4
            b1 = f(b0, svm.w, svm.b, -1)
            ax.plot([a0, b0], [a1, b1], 'k--')

            ax.scatter(sample.transpose()[:, 0], sample.transpose()[:, 1], c=clazzes, marker='.', cmap='tab20',
                        label="эл-ты выборки")
            ax.scatter(svm.sv[:, 0], svm.sv[:, 1], c=svm.sv_y, marker='*', cmap='tab20',
                        label="соб. в-ры")
            ax.legend()
            ax.grid(True)
            print(f"w={svm.w} b={svm.b}")
            print(f"2 / np.linalg.norm(svm.w)={2 / np.linalg.norm(svm.w)}")
            print("--------")

        mean1 = (0, 0)
        mean2 = (2, -2)
        cov = [[1, 0.3],
               [0.3, 1]]
        size1, size2 = 100, 100
        sample1 = np.random.multivariate_normal(mean1, cov, size1).transpose()
        sample2 = np.random.multivariate_normal(mean2, cov, size2).transpose()
        sample = np.concatenate((sample1, sample2), axis=1)
        clazzes = np.zeros(size1 + size2)
        for i in range(size1):
            clazzes[i] = 1
        for i in range(size1, size1 + size2):
            clazzes[i] = -1

        c_list = [1, 10, 100, 1000]
        fig, axs = plt.subplots(2, 2)
        for i, c in enumerate(c_list):
            svm = SVM()
            svm.fit(sample.transpose(), clazzes, C=c)
            ax = axs[i // 2, i % 2]
            plot()
            predicted_classes = svm.predict(sample.transpose())
            ax.set_title(f"C={c} errors = {np.count_nonzero(predicted_classes != clazzes)} M={2 / np.linalg.norm(svm.w)}")
            # print(f"errors = {np.count_nonzero(predicted_classes != clazzes)}")

        plt.show()

    @staticmethod
    def part3():
        mean1 = (0, 8)
        mean2 = (0, 2)
        cov1 = [[4, 0],
               [0, 2]]
        cov2 = [[4, 0],
               [0, 2]]
        size1, size2 = 100, 100
        sample1 = np.random.multivariate_normal(mean1, cov1, size1)
        sample2 = np.random.multivariate_normal(mean2, cov2, size2)
        sample = np.concatenate((sample1, sample2), axis=0)

        c_sample = sample.copy()
        for x, y in zip(sample, c_sample):
            y[0] = x[0]
            y[1] = x[1] + fabs(x[1] * x[0] * x[0])
        clazzes = np.zeros(size1 + size2)
        for i in range(size1):
            clazzes[i] = 1
        for i in range(size1, size1 + size2):
            clazzes[i] = -1
        plt.scatter(c_sample[:, 0], c_sample[:, 1], c=clazzes, marker='.', cmap='rainbow',
                    label="эл-ты выборки")
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    Lab3.part1()
    Lab3.part2()
    # Lab3.part3()