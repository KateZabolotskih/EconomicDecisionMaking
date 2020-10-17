import numpy as np
import os
import Lab1.BayesianClassification as BC
root_path = os.path.abspath("../")
data_path = os.path.join(root_path, 'data/german.data')
if not os.path.exists(data_path):
    print(f"cannot find data file: {data_path}")
    exit(1)

cov = [[7, 3,  2],
       [3, 10, 5],
       [2, 5,  8]]

mean1 = (5, 5, 5)
mean2 = (-5, 5, 5)
X = np.random.multivariate_normal(mean1, cov, 110).transpose()
Y = np.random.multivariate_normal(mean2, cov, 90).transpose()

bc = BC.BayesianClassification(Y, X)

P = np.random.multivariate_normal(mean1, cov, 510).transpose()
W = np.random.multivariate_normal(mean2, cov, 500).transpose()

bc.classify(X, Y)
bc.classify(P, W)
#bc.plot_data_3D(X, Y)
#bc.plot_data_3D(P, W)

print('P_21 = ' + str(bc.P_21) + '\n'
      'P_12 = ' + str(bc.P_12) + '\n'
      'r1 = ' + str(bc.r1) + '\n'
      'r2 = ' + str(bc.r2) + '\n'
      'D = ' + str(bc.D) + '\n'
      'DH = ' + str(bc.DH) + '\n')


#p = 5
#col1 = 0
#col2 = 2
#col3 = 9
#col4 = 11
#col5 = 20

p = 4
col1 = 1
col2 = 3
col3 = 9
col4 = 19

matrix = np.loadtxt(data_path)

 # выбираем из матрицы "объект-свойсво" столбцы (признаки) в количесве p штук)
 # составляем обучающую или тестовую матрицу таким образом
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

# составляем обучающие выборки
n1 = 100
sample1 = get_sample_from_data(matrix, [col1, col2, col3, col4], n1, marker=2, p=p)
n2 = 90
sample2 = get_sample_from_data(matrix, [col1, col2, col3, col4], n2, marker=1, p=p)
# "обучаем" классификатор
BC = BC.BayesianClassification(sample1, sample2)
print('training samples:')
BC.classify(sample1, sample2)

# составляем тестовые выборки
n3 = 400
sample3 = get_sample_from_data(matrix, [col1, col2, col3, col4], n3, marker=2, p=p, start_from=20)
n4 = 300
sample4 = get_sample_from_data(matrix, [col1, col2, col3, col4], n4, marker=1, p=p, start_from=20)
# используем уже "обученный" классификатор для разделения данных тестовой выборки
print('testing samples:')
BC.classify(sample3, sample4)

# дынные обученного классификатора
print('characteristics of classifier:')
print('P_21 = ' + str(BC.P_21) + '\n'
      'P_12 = ' + str(BC.P_12) + '\n'
      'r1 = ' + str(BC.r1) + '\n'
      'r2 = ' + str(BC.r2) + '\n'
      'D = ' + str(BC.D) + '\n'
      'DH = ' + str(BC.DH) + '\n')

