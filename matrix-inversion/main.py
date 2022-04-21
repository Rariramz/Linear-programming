import numpy as np


def input_matrix(n, message):
    print(message)
    return [[float(j) for j in input().split()] for i in range(n)]


def input_vector(message):
    print(message)
    return list(map(int, input().split()))


def matrix_inversion(n, i, matrix_a, matrix_a_inverse, vector_x, logger=print):
    def log(message):
        if logger:
            logger(message)

    i -= 1
    # A2 - матрица, полученная из A заменой i-го столбца на столбец x
    # A2_inverse - обратная матрица A2

    log('\n\tМатрица A:')
    log(matrix_a)
    log('\tОбратная матрица A:')
    log(matrix_a_inverse)
    log('\tВектор x(T):')
    log(vector_x)
    log(f'\ti = {i + 1}')

    log('\nШАГ 1: находим l0 = A_inverse * x:')
    vector_l0 = matrix_a_inverse.dot(vector_x)
    log(vector_l0)

    log('\nЕсли i-й элемент вектора l0[i] == 0, то матрица A2, '
          'полученная из A заменой i-го столбца на столбец x, необратима - конец метода.')
    log('Матрица обратима?')
    log(f'l0[i] = {vector_l0[i]}')
    if vector_l0[i] == 0:
        return 'НЕТ'
    log('ДА')

    log('\nШАГ 2: формируем вектор l1, который получается из вектора l0 заменой i-го элемента на -1:')
    vector_l1 = vector_l0.copy()
    vector_l1[i] = -1
    log(vector_l1)

    log('\nШАГ 3: находим l2 = -1 / l0[i] * l1:')
    vector_l2 = (-1 / vector_l0[i]) * vector_l1
    log(vector_l2)

    log('\nШАГ 4: формируем матрицу Q, получающуюся из единичной матрицы порядка n заменой i-го столбца на столбец l2:')
    matrix_q = np.eye(n)
    matrix_q[:, i] = vector_l2
    log(matrix_q)

    log('\nШАГ 5: находим A2_inverse = Q * A_inverse:')
    return matrix_q.dot(matrix_a_inverse)


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    n_ = int(input('Размерность матрицы n = '))
    i_ = int(input(f'Номер столбца, который нужно заменить (1 < i <= {n_}) i = '))
    a = np.array(input_matrix(n_, 'Введите матрицу A, отделяя строки матрицы переводом строки:'))
    a_inverse = np.array(input_matrix(n_, 'Введите обратную матрицу A, отделяя строки матрицы переводом строки:'))
    x = np.array(input_vector('Введите вектор x(T):'))

    print(matrix_inversion(n_, i_, a, a_inverse, x))
