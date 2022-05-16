import numpy as np


def input_float_matrix(n, message=""):
    print(message)
    return np.array([[float(j) for j in input().split()] for _ in range(n)])


def input_float_vector(message=""):
    print(message)
    return list(float(_) for _ in input().split())


def matrix_multiplication(matrix_a, matrix_b, index):
    result_matrix = np.array([[0 for _ in range(len(matrix_a[0]))] for _ in range(len(matrix_a))])
    for i in range(len(matrix_a)):
        for j in range(len(matrix_a[0])):
            for k in range(len(matrix_a[0])):
                if not (k == index or k == i):
                    continue
                result_matrix[i, j] += matrix_a[i, k] * matrix_b[k, j]
    return result_matrix


def matrix_inversion(n, i, matrix_a_inverse, vector_x, logger=print):
    def log(message):
        if logger:
            logger(message)

    log('~~~~~~ Обращение матрицы с измененным столбцом ~~~~~')
    # A2 - матрица, полученная из A заменой i-го столбца на столбец x
    # A2_inverse - обратная матрица A2

    log('\tОбратная матрица A:')
    log(matrix_a_inverse)
    log('\tВектор x(T):')
    log(vector_x)
    log(f'\ti = {i + 1}')

    log('\nШАГ 1: находим l0 = A_inverse * x:')
    vector_l0 = matrix_a_inverse.dot(vector_x)
    log(f'\tl0 = {vector_l0}')

    log('\nЕсли i-й элемент вектора l0[i] == 0, то матрица A2, ')
    log('полученная из A заменой i-го столбца на столбец x, необратима - конец метода.')
    log('Матрица обратима?')
    log(f'\tl0[i] = {vector_l0[i]}')
    if vector_l0[i] == 0:
        log('\tНЕТ')
        return None
    log('\tДА')

    log('\nШАГ 2: формируем вектор l1, который получается из вектора l0 заменой i-го элемента на -1:')
    vector_l1 = vector_l0.copy()
    vector_l1[i] = -1
    log(f'\tl1 = {vector_l1}')

    log('\nШАГ 3: находим l2 = -1 / l0[i] * l1:')
    vector_l2 = (-1 / vector_l0[i]) * vector_l1
    log(f'\tl2 = {vector_l2}')

    log('\nШАГ 4: формируем матрицу Q, получающуюся из единичной матрицы порядка n')
    log('заменой i-го столбца на столбец l2:')
    matrix_q = np.eye(n)
    matrix_q[:, i] = vector_l2
    log(f'\tQ =\n{matrix_q}')

    log('\nШАГ 5: находим A2_inverse = Q * A_inverse:')
    matrix_a2_inverse = matrix_multiplication(matrix_q, matrix_a_inverse, i)
    log(f'\tA2 =\n{matrix_a2_inverse}')
    return matrix_a2_inverse


if __name__ == '__main__':
    np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})

    # n_ = int(input('Размерность матрицы n = '))
    # i_ = int(input(f'Номер столбца, который нужно заменить (1 < i <= {n_}) i = '))
    # a = input_float_matrix(n_, 'Введите матрицу A, отделяя строки матрицы переводом строки:')
    # a_inverse = input_float_matrix(n_, 'Введите обратную матрицу A, отделяя строки матрицы переводом строки:')
    # x = input_float_vector('Введите вектор x(T):')

    # Тестовый случай
    n_ = 3
    i_ = 3
    a = np.array([[1, -1, 0],
                  [0, 1, 0],
                  [0, 0, 1]])
    a_inverse = np.array([[1, 1, 0],
                          [0, 1, 0],
                          [0, 0, 1]])
    x = list([1, 0, 1])

    matrix_inversion(n_, i_ - 1, a_inverse, x)
