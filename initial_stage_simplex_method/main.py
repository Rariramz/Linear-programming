import numpy as np
from matrix_inversion import input_float_vector, input_float_matrix
from main_stage_simplex_method import main_stage_simplex_method, create_matrix_ab, input_int_vector


def initial_stage_simplex_method(matrix_a, vector_b, m, n, logger=print):
    def log(message):
        if logger:
            logger(message)

    log('~~~~~~ Начальная фаза симплекс-метода ~~~~~')
    log('\tМатрица A:')
    log(matrix_a)
    log('\tВектор b(T):')
    log(vector_b)
    log(f'\tm = {m}')
    log(f'\tn = {n}')

    log('\nШАГ 1: преобразуем задачу так, чтобы вектор правых частей b был неотрицательным:')
    for i in range(m):
        if vector_b[i] < 0:
            vector_b[i] *= -1
            matrix_a[i] *= -1
    log(f'\tb = {vector_b}')
    log(f'\tA = \n{matrix_a}')

    log('\nШАГ 2: Составим вспомогательную задачу, где c1 = ( 0 x n, (-1) x m ), ')
    log('а матрица A1 получается из A присоединением к ней справа E порядка m:')
    zeros = [0. for i in range(n)]
    ones = [-1. for i in range(m)]
    matrix_a1 = np.concatenate((matrix_a, np.eye(m)), axis=1)
    log(f'\tA1 = \n{matrix_a1}')
    vector_c1 = np.array(zeros+ones)
    log(f'\tc1 = {vector_c1}')

    log('\nШАГ 3: Построим начальный базисный допустимый план задачи, ')
    log('где x1 = (0 x n, b1, b2, .., bm) и B = { j1=n+1, j2=n+2, ..., jm=n+m }:')
    vector_x1 = np.array(zeros + vector_b.copy())
    vector_jb = [i for i in range(n, n + m)]
    log(f'\tx1 = {vector_x1}')
    log(f'\tB = {list(_ + 1 for _ in vector_jb)}')

    log('\nШАГ 4: Решим вспомогательную задачу основной фазой симплекс-метода и получим оптимальный план:')
    vector_x1, vector_jb = main_stage_simplex_method(
        m, n + m, matrix_a1, vector_x1, vector_c1, vector_jb, False)
    log(f'\tx1 = {vector_x1}')
    log(f'\tB = {list(_ + 1 for _ in vector_jb)}')

    log('\nШАГ 5: Проверим условия совместности: искусственные x равны 0 => задача совместна:')
    vector_x1_original = vector_x1[:n]
    vector_x1_fake = vector_x1[n:]
    log(f'\tx1_fake = {vector_x1_fake}')
    for i in range(len(vector_x1_fake)):
        if vector_x1_fake[i] != 0:
            return 'Задача не совместна'
    log('\tЗадача совместна! :)')

    log('\nШАГ 6: Формируем допустимый план x = ( x1(1), x2(2), ..., x1(n) ) задачи (1). ')
    log('Для него необходимо подобрать множество B.')
    vector_x = vector_x1_original
    log(f'\tx = {vector_x}')

    def from_seven_step(matrix_a, vector_jb, vector_b, matrix_a1):
        log('\nШАГ 7: Если множество B состоит только из индексов неискусственных переменных, ')
        log('то метод завершает свою работу и возвращает базисный допустимый план (x, B).')
        log(f'\tB = {list(_ + 1 for _ in vector_jb)}')
        is_only_original = True
        for i in range(len(vector_jb)):
            if vector_jb[i] >= n:
                is_only_original = False
        if is_only_original:
            log(f'\nОтвет:\n\t(x(T) = ({" ".join(map(str, vector_x))}), '
                f'B = ({", ".join(map(str, list(_ + 1 for _ in vector_jb)))})) - базисный допустимый план.')
            return vector_x, vector_jb

        log('\nШАГ 8: Выберем в наборе B максимальный индекс искусственной переменной jk = n + i:')
        vector_jb_fake = np.zeros(m)
        for i in range(len(vector_jb)):
            if vector_jb[i] >= n:
                vector_jb_fake[i] = i
        k = int(max(vector_jb_fake))
        jk = vector_jb[k]
        _i = jk - n
        log(f'\tjk = {jk + 1} = {n} + {i + 1}, k = {k + 1}, i = {i + 1}')

        log('\nШАГ 9: Для каждого небазисного индекса вычислим вектор l(j) = A1b_inverse * A1[j]:')
        vector_j = {_ for _ in range(0, n)}
        vector_j_not_b = [_ for _ in vector_j if _ not in vector_jb]
        list_vectors_l = np.zeros((len(vector_j_not_b), len(vector_j_not_b)))
        matrix_a1b = create_matrix_ab(matrix_a1.copy(), vector_jb.copy())
        matrix_a1b_inverse = np.linalg.inv(matrix_a1b.copy())
        for j in range(len(vector_j_not_b)):
            list_vectors_l[j] = matrix_a1b_inverse.dot(matrix_a1[:, vector_j_not_b[j]])
            log(f'\tvector_l({j + 1}) = {list_vectors_l[j]}')

        log('\nШАГ 10: Если найдется небазисный индекс j такой, что l(j)k != 0, '
            'то заменим в наборе B значение jk на j:')
        for j in range(len(vector_j_not_b)):
            vector_l = list_vectors_l[j]
            if vector_l[k] != 0:
                vector_jb[k] = j
        log(f'\tB = {list(_ + 1 for _ in vector_jb)}')

        log('\nШАГ 11: Если для каждого небазисного индекса j выполняется (l(j))k = 0, ')
        log('то i-е основное ограничение задачи (1) излишне, его необходимо удалить.')
        is_odd = True
        for j in range(len(vector_j_not_b)):
            vector_l = list_vectors_l[j]
            if vector_l[k] != 0:
                is_odd = False
        if is_odd:
            matrix_a = np.delete(matrix_a, _i, 0)
            vector_b = np.delete(vector_b, _i, 0)
            vector_jb = np.delete(vector_jb, k)
            matrix_a1 = np.delete(matrix_a1, _i, 0)
            log(f'\tA = \n{matrix_a}')
            log(f'\tb = {vector_b}')
            log(f'\tB = {list(_ + 1 for _ in vector_jb)}')
            log(f'\tA1 = \n{matrix_a1}')

        log('\nПереходим на ШАГ 7.')
        from_seven_step(matrix_a, vector_jb, vector_b, matrix_a1)

    return from_seven_step(matrix_a, vector_jb, vector_b, matrix_a1)


def simplex():
    # m_ = int(input('m = '))
    # n_ = int(input('n = '))
    # a = input_float_matrix(m_, 'Введите матрицу A, отделяя строки матрицы переводом строки:')
    # b = input_int_vector('Введите вектор b(T):')
    # c = input_int_vector('Введите вектор c(T):')

    # Тестовый пример
    m_ = 2
    n_ = 3
    a = np.array([[1, 1, 1],
                  [2, 2, 2]])
    b = list([0, 0])
    c = list([1, 0, 0])

    initial_stage_simplex_method(a, b, m_, n_)


if __name__ == "__main__":
    simplex()
