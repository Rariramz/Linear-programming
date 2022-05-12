import numpy as np
from matrix_inversion import matrix_inversion, input_float_vector, input_float_matrix


def input_int_vector(message):
    print(message)
    return list(int(j) for j in input().split())


def create_matrix_ab(matrix_a, vector_jb):
    # ШАГ 1: строим базисную матрицу Ab, состоящую из столбцов матрицы A с индексами из B:
    n = len(vector_jb)
    matrix_ab = np.zeros((n, n))
    for i, ib in enumerate(vector_jb):
        matrix_ab[:, i] = matrix_a[:, ib]
    return matrix_ab


def create_vector_cb(vector_c, vector_jb):
    # ШАГ 2: формируем вектор cb - вектор компонент вектора c, чьи индексы принадлежат множеству B:
    vector_cb = np.zeros(len(vector_jb))
    for i, ib in enumerate(vector_jb):
        vector_cb[i] = vector_c[ib]
    return vector_cb


def find_min_delta(vector_delta, vector_j, vector_jb):
    # ШАГ 5: проверим условие оптимальности текущего плана x, а именно,
    # если delta >= 0, то текущий x является оптимальным планом - конец метода.
    # (проверяем небазисные индексы):
    vector_j_not_b = vector_j - set(vector_jb)
    if vector_j_not_b == set():
        return (0, 0)
    delta_vals = []
    for i in vector_j_not_b:
        delta_vals.append(vector_delta[i])
    delta_min = min(delta_vals)
    return delta_vals.index(delta_min), delta_min


def create_vector_theta(vector_x, vector_z, vector_jb):
    # ШАГ 8: находим вектор theta(T) = (theta1, .., thetam) по правилу
    # theta[i] = z[i] > 0 ? x[j[i]] / z[i] : infinity,
    # где j[i] — i-й по счету базисный индекс в упорядоченном наборе B.
    vector_theta = []
    for z, jb in zip(vector_z, vector_jb):
        if z > 0:
            theta = vector_x[jb] / z
            vector_theta.append(theta)
        else:
            vector_theta.append(float('inf'))
    return vector_theta


def create_vector_x(vector_x, vector_z, vector_jb, j0, theta_min, m):
    vector_x_new = np.zeros_like(vector_x)
    for i in range(m):
        index = vector_jb[i]
        # ШАГ 13: обновим компоненты плана x: x[j0] = theta_min и для каждого i != k: x[j[i]] -= theta
        vector_x_new[index] = vector_x[index] - theta_min * vector_z[i]
    vector_x_new[j0] = theta_min
    return vector_x_new


def main_stage_simplex_method(m, n, matrix_a, vector_x, vector_c, vector_jb, logger=print):
    def log(message):
        if logger:
            logger(message)

    log('\tМатрица A:')
    log(matrix_a)
    log('\tВектор x(T):')
    log(vector_x)
    log('\tВектор c(T):')
    log(vector_c)
    log('\tВектор B:')
    log(list(j + 1 for j in vector_jb))

    log('~~~~~ Основная фаза симплекс-метода ~~~~~')
    vector_j = {i for i in range(0, n)}  # Множество индексов

    iteration = 1
    while True:
        log(f'\n~~~~~~~~~~~~~~~~~~~~ ИТЕРАЦИЯ #{iteration} ~~~~~~~~~~~~~~~~~~~~')
        log('\nШАГ 1: строим базисную матрицу Ab, состоящую из столбцов матрицы A с индексами из B:')
        matrix_ab = create_matrix_ab(matrix_a.copy(), vector_jb.copy())
        log(matrix_ab)
        try:
            log('\nНаходим обратную матрицу Ab_inverse:')
            if iteration == 1:
                matrix_ab_inverse = np.linalg.inv(matrix_ab.copy())
            else:
                matrix_ab_inverse = matrix_inversion(len(vector_jb), k, matrix_ab_inverse, matrix_a[:, j0])
            log(matrix_ab_inverse)
        except np.linalg.LinAlgError:
            log('..не удалось построить Ab_inverse.')
            log('\nОтвет: целевой функционал задачи не ограничен сверху на множестве допустимых планов.')
            exit()

        log('\nШАГ 2: формируем вектор cb - вектор компонент вектора c, чьи индексы принадлежат множеству B:')
        vector_cb = create_vector_cb(vector_c.copy(), vector_jb.copy())
        log(f'\tcb = {vector_cb}')

        log('\nШАГ 3: находим вектор потенциалов u(T) = cb(T) * Ab_inverse:')
        vector_u = vector_cb.dot(matrix_ab_inverse.copy())
        log(f'\tu(T) = {vector_u}')

        log('\nШАГ 4: находим вектор оценок delta(T) = u(T) * A - c(T):')
        vector_delta = vector_u.dot(matrix_a.copy()) - vector_c
        log(f'\tdelta(T) = {vector_delta}')

        log('\nШАГ 5: проверим условие оптимальности текущего плана x, а именно,')
        log('\tесли delta >= 0, то текущий x является оптимальным планом - конец метода.')
        log('\tТекущий план является оптимальным планом?')
        j0, min_delta = find_min_delta(vector_delta, vector_j, vector_jb)
        log(f'\tmin_delta = {min_delta}')
        if min_delta >= 0:
            log('\tДА')
            return f'\nОтвет:\n\t(x(T) = ({" ".join(map(str, vector_x))}), ' \
                   f'B = ({", ".join(map(str, vector_jb))})) - оптимальный план.'
        log('\tНЕТ')

        log('\nШАГ 6: находим в векторе оценок delta первую отрицательную компоненту и сохраняем её индекс в j0:')
        j0 = vector_delta.argmin()
        log(f'\tj0 = {j0 + 1}')

        log('\nШАГ 7: вычислим вектор z = Ab_inverse * A[j0](столбец j0 матрицы A):')
        vector_z = matrix_ab_inverse.dot(matrix_a[:, j0])
        log(f'\tz(T) = {vector_z}')

        log('\nШАГ 8: находим вектор theta(T) = (theta1, .., thetam) по правилу')
        log('\ttheta[i] = z[i] > 0 ? x[j[i]] / z[i] : infinity,')
        log('\tгде j[i] — i-й по счету базисный индекс в упорядоченном наборе B.')
        vector_theta = create_vector_theta(
            vector_x.copy(), vector_z.copy(), vector_jb.copy())
        log(f'\ttheta(T) = {vector_theta}')

        log('\nШАГ 9: вычислим theta_min вектора theta:')
        theta_min = min(vector_theta)
        log(f'\ttheta_min = {theta_min}')

        log('\nШАГ 10: проверяем условие неограниченности целевого функционала:')
        log('\tесли theta_min = infinity, то конец метода.')
        log('\ttheta_min = infinity ?')
        if theta_min == float('inf'):
            log('\tДА')
            return '\nОтвет: целевой функционал задачи не ограничен сверху на множестве допустимых планов.'
        log('\tНЕТ')

        log('\nШАГ 11: находим первый индекс k, на котором достигается минимум theta,')
        log('\tсохраняем в jbk k-й базисный индекс из B:')
        k = vector_theta.index(theta_min)
        jbk = vector_jb[k]
        log(f'\tk = {k + 1}')
        log(f'\tjbk = {jbk + 1}')

        log('\nШАГ 12: в упорядоченном множестве B заменим k-й индекс jbk на индекс j0:')
        vector_jb[vector_jb.index(jbk)] = j0
        log(f'\tB = {list(i + 1 for i in vector_jb)}')

        log('\nШАГ 13: обновим компоненты плана x: x[j0] = theta_min и для каждого i != k: x[j[i]] -= theta:')
        vector_x = create_vector_x(
            vector_x.copy(), vector_z.copy(), vector_jb.copy(),
            j0, theta_min, m)
        log(f'\tx(T) = {vector_x}')
        iteration += 1


if __name__ == '__main__':
    # m_ = int(input('m = '))
    # n_ = int(input('n = '))
    # a = input_float_matrix(m_, 'Введите матрицу A, отделяя строки матрицы переводом строки:')
    # c = input_int_vector('Введите вектор c(T):')
    # x = input_float_vector('Введите вектор x(T):')
    # jb_ = input_int_vector('Введите множество базисных индексов jb:')

    # Тестовый пример
    m_ = 3
    n_ = 5
    a = np.array([[-1, 1, 1, 0, 0],
                  [1, 0, 0, 1, 0],
                  [0, 1, 0, 0, 1]])
    c = list([1, 1, 0, 0, 0])
    x = list([0, 0, 1, 3, 2])
    jb_ = list([3, 4, 5])

    jb = list(j - 1 for j in jb_)
    print(main_stage_simplex_method(m_, n_, a, x, c, jb))
