import numpy as np
from matrix_inversion import input_float_vector, input_float_matrix
from main_stage_simplex_method import create_matrix_ab, create_vector_cb, input_int_vector


def dual_simplex_method(m, n, matrix_a, vector_b, vector_c, vector_jb, logger=print):
    def log(message):
        if logger:
            logger(message)

    log('~~~~~~ Двойственный симплекс-метод ~~~~~')
    log('\tМатрица A:')
    log(matrix_a)
    log('\tВектор b(T):')
    log(vector_b)
    log('\tВектор c(T):')
    log(vector_c)
    log('\tМножество базисных индексов B:')
    log(list(_ + 1 for _ in vector_jb))
    log(f'\tm = {m}')
    log(f'\tn = {n}')

    iteration = 1
    while True:
        log(f'\n~~~~~~~~~~~~~~~~~~~~ ИТЕРАЦИЯ #{iteration} ~~~~~~~~~~~~~~~~~~~~')
        log('\nШАГ 1: Составим матрицу Ab и найдем Ab_inverse:')
        matrix_ab = create_matrix_ab(matrix_a, vector_jb)
        matrix_ab_inverse = np.linalg.inv(matrix_ab)
        log(f'\tAb =\n{matrix_ab}')
        log(f'\tAb_inverse =\n{matrix_ab_inverse}')

        log('\nШАГ 2: Сформируем вектор cb:')
        vector_cb = create_vector_cb(vector_c, vector_jb)
        log(f'\tcb(T) = {vector_cb}')

        log('\nШАГ 3: Находим базисный допустимый план двойственной задачи y(T) = cb(T) * Ab_inverse:')
        vector_y = vector_cb.dot(matrix_ab_inverse)
        log(f'\ty(T) = {vector_y}')

        log('\nШАГ 4: Находим псевдоплан K(T) = (Kb, Kn), соответствующий текущему баз. доп. плану; ')
        log('Kb = Ab_inverse * b, Kn = 0:')
        vector_k = np.zeros(n)
        vector_kb = matrix_ab_inverse.dot(vector_b)
        for i in range(len(vector_kb)):
            index = vector_jb[i]
            vector_k[index] = vector_kb[i]
        log(f'\tK(T) = {vector_k}')

        log('\nШАГ 5: Если K >= 0, то K - оптимальный план прямой задачи (1) и метод завершает работу.')
        log('K >= 0?')
        if min(vector_k) >= 0:
            log('\tДА')
            log(f'\nОтвет:\n\t(x(T) = ({" ".join(map(str, vector_k))}), '
                f'B = ({", ".join(map(str, list(_ + 1 for _ in vector_jb)))})) ')
            log(f'\t- оптимальный план прямой задачи (1).')
            return vector_k, vector_jb
        else:
            log('\tНЕТ')

        log('\nШАГ 6: Выделим отрицательную компоненту псевдоплана K и сохраним её индекс в переменной k:')
        k_min = min(list(vector_k))
        jk = int(list(vector_k).index(k_min))
        k = int(vector_jb.index(jk))
        log(f'\tK_min = {k_min}, jk = {jk + 1}, k = {k + 1}')

        log('\nШАГ 7: Пусть delta_y - это k-я строка матрицы Ab_inverse. ')
        log('Для каждого небазисного j вычислим m[j] = delta_y(T) * A[j] (A[j] - j-й столбец A):')
        delta_y = matrix_ab_inverse[k]
        vector_j = {_ for _ in range(0, n)}
        vector_j_not_b = list(vector_j - set(vector_jb))
        vector_m = np.zeros(len(vector_j_not_b))
        for j in range(len(vector_m)):
            vector_m[j] = delta_y.dot(matrix_a[:, vector_j_not_b[j]])
            log(f'\tm({vector_j_not_b[j] + 1}) = {vector_m[j]}')

        log('\nШАГ 8: Если для каждого небазисного j m[j] >= 0, то прямая задача (1) не совместна.')
        log('Задача совместна?')
        log(f'\tНебазисные индексы: {list(_ + 1 for _ in vector_j_not_b)}')
        vector_j_not_b_m_less_0 = []
        isOkay = False
        for i in range(len(vector_m)):
            if vector_m[i] < 0:
                isOkay = True
                vector_j_not_b_m_less_0.append(list(vector_j_not_b)[i])
        if isOkay:
            log('\tДА')
        else:
            log('\tНЕТ')
            return None

        log('\nШАГ 9: Для каждого небазисного j & m[j] < 0 вычислим s[j] = (c[j] - A[j](T) * y) / m[j]:')
        vector_s = np.zeros(len(vector_j_not_b_m_less_0))
        for i in range(len(vector_s)):
            j = int(vector_j_not_b_m_less_0[i])
            vector_s[i] = (vector_c[j] - matrix_a[:, j].dot(vector_y)) / vector_m[i]
            log(f'\ts({vector_j_not_b[i] + 1}) = {vector_s[i]}')

        log('\nШАГ 10: Найдём s0 = smin и сохраним индекс, на котором достигнут min в переменной j0:')
        s0 = min(vector_s)
        j0 = vector_j_not_b[list(vector_s).index(s0)]
        log(f'\ts0 = {s0}, j0 = {j0 + 1}')

        log('\nШАГ 11: В множестве базисных индексов B заменим k-й базисный индекс на j0:')
        vector_jb[k] = j0
        log(f'\tB = {list(_ + 1 for _ in vector_jb)}')
        iteration += 1


def simplex():
    # m_ = int(input('m = '))
    # n_ = int(input('n = '))
    # a = input_float_matrix(m_, 'Введите матрицу A, отделяя строки матрицы переводом строки:')
    # b = input_int_vector('Введите вектор b(T):')
    # c = input_int_vector('Введите вектор c(T):')
    # jb_ = input_int_vector('Введите множество базисных индексов B нач. баз. доп. плана дв. задачи:')

    # Тестовый пример
    m_ = 2
    n_ = 5
    a = np.array([[-2, -1, -4, 1, 0],
                  [-2, -2, -2, 0, 1]])
    b = list([-1, -3/2])
    c = list([-4, -3, -7, 0, 0])
    jb_ = list([4, 5])

    jb = list(_ - 1 for _ in jb_)
    result = dual_simplex_method(m_, n_, a, b, c, jb)
    if not result:
        print('\tНет решений.')


if __name__ == "__main__":
    simplex()
