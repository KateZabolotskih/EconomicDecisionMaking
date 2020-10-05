import numpy as np
import os
import Lab1.BayesianClassification as BC
root_path = os.path.abspath("../")
data_path = os.path.join(root_path, 'data/german.data')
if not os.path.exists(data_path):
    print(f"cannot find data file: {data_path}")
    exit(1)

p = 3
col1 = 2
col2 = 3
col3 = 8

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
n1 = 300
sample1 = get_sample_from_data(matrix, [col1, col2, col3], n1, marker=1, p=p)
n2 = 300
sample2 = get_sample_from_data(matrix, [col1, col2, col3], n2, marker=2, p=p)
# "обучаем" классификатор
BC = BC.BayesianClassification(sample1, sample2)
print('training samples:')
BC.plot_data(sample1, sample2)

# составляем тестовые выборки
n3 = 50
sample3 = get_sample_from_data(matrix, [col1, col2, col3], n3, marker=1, p=p, start_from=20)
n4 = 50
sample4 = get_sample_from_data(matrix, [col1, col2, col3], n4, marker=2, p=p, start_from=20)
# используем уже "обученный" классификатор для разделения данных тестовой выборки
print('testing samples:')
BC.plot_data(sample3, sample4)

# дынные обученного классификатора
print('characteristics of classifier:')
print('P_21 = ' + str(BC.P_21) + '\n'
      'P_12 = ' + str(BC.P_12) + '\n'
      'q1 = ' + str(BC.q1) + '\n'
      'q2 = ' + str(BC.q2) + '\n'
      'D = ' + str(BC.D) + '\n'
      'DH = ' + str(BC.DH) + '\n')

