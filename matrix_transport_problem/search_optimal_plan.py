import numpy as np


def search_optimal_plan(matrix_x, array_b, matrix_c, logger):
    def log(message):
        if logger:
            logger(message)

    m = len(matrix_c)
    n = len(matrix_c[0])
    len_b = len(array_b)

    # уравнения под каждую базисную клетку + свободная переменная
    equations_count = len_b + 1
    # переменные vector_u - m штук, vector_v - n штук
    variables_count = m + n

    iteration = 0
    while True:
        # matrix_m - матрица коэффициентов системы лин. уравнений
        matrix_m = np.zeros((equations_count, variables_count))
        for i in range(len_b):
            u_index, v_index = array_b[i]     # хранятся значения с 0!
            matrix_m[i][u_index] = 1
            matrix_m[i][m + v_index] = 1
        # уравнение для свободной переменной B[0][0] (первый попавшийся индекс в В)
        matrix_m[-1][array_b[0][0]] = 1

        # vector_d - вектор свободных членов системы лин. уравнений
        vector_d = np.zeros(equations_count)
        for k in range(len_b):
            i = array_b[k][0]
            j = array_b[k][1]
            vector_d[k] = matrix_c[i][j]

        uv = np.linalg.solve(matrix_m, vector_d)
        vector_u = uv[:m]
        vector_v = uv[m:]

        iteration += 1
        log(f'\n~~~~~~~~~~~~~~~~ ИТЕРАЦИЯ {iteration} ~~~~~~~~~~~~~~~~~~~~~')
        log('\nСистема лин. уравнений для нахождения потенциалов: matrix_m = [uv] = vector_d')
        log(f'M = \n{matrix_m}')
        log(f'd = {vector_d}')
        log(f'u = {vector_u}')
        log(f'v = {vector_v}')

        array_not_b = []
        for i in range(m):
            for j in range(n):
                if (i, j) not in array_b:
                    array_not_b.append((i, j))

        log(f'Для небазисных ~В = {[(_b[0]+1, _b[1]+1) for _b in array_not_b]}')

        is_x_optimal = True
        for (i, j) in array_not_b:
            log(f'u[{i + 1}] + v[{j + 1}] = {vector_u[i] + vector_v[j]} {">" if vector_u[i] + vector_v[j] > matrix_c[i][j] else "<="} {matrix_c[i][j]} = C[{i + 1}][{j + 1}]')
            if vector_u[i] + vector_v[j] > matrix_c[i][j]:
                log(f'\tДобавляем в В {(i + 1, j + 1)}')
                first_bad = (i, j)
                is_x_optimal = False
                break
        if is_x_optimal:
            return matrix_x, array_b

        array_b.append(first_bad)

        matrix_gb = array_b.copy()
        # row_matrix_gb, col_matrix_gb - индексы из matrix_gb,
        # разбитые на 2 массива (если индексов строки в row_matrix_gb нечетное кол-во
        # или индекса строки там нет - удаляем строку; аналогично для столбцов)
        row_matrix_gb, col_matrix_gb = list(zip(*matrix_gb))
        row_matrix_gb, col_matrix_gb = list(row_matrix_gb), list(col_matrix_gb)

        while True:
            del_row = False
            del_col = False

            for i in range(m):
                # если 0 - ничего не надо, его и так нет
                if row_matrix_gb.count(i) == 1:
                    delete_index = row_matrix_gb.index(i)
                    del matrix_gb[delete_index]
                    del row_matrix_gb[delete_index]
                    del col_matrix_gb[delete_index]
                    del_row = True

            for i in range(n):
                if col_matrix_gb.count(i) == 1:
                    delete_index = col_matrix_gb.index(i)
                    del matrix_gb[delete_index]
                    del row_matrix_gb[delete_index]
                    del col_matrix_gb[delete_index]
                    del_col = True

            if not del_row and not del_col:
                break

        matrix_gb_copy = matrix_gb.copy()
        matrix_gb_linked = list()
        matrix_gb_linked.append(matrix_gb_copy.pop())
        row_matrix_gb.pop()
        col_matrix_gb.pop()

        while len(matrix_gb_copy) != 1:
            # Цель: добавить в matrix_gb_linked следующий эл-т с i как у предыдущего
            i_prev = matrix_gb_linked[-1][0]
            # в массиве i-шек ищем совпадающую с i_prev
            next_el_index = row_matrix_gb.index(i_prev)
            # переносим эл-т с найденным индексом из matrix_gb_copy в matrix_gb_linked
            matrix_gb_linked.append(matrix_gb_copy.pop(next_el_index))
            # удаляем из массивов i-шек и j-шек перенесенный эл-т, чтобы было соотв. индексов с matrix_gb_copy
            row_matrix_gb.pop(next_el_index)
            col_matrix_gb.pop(next_el_index)

            # Цель: добавить в matrix_gb_linked следующий эл-т с j как у предыдущего
            j_prev = matrix_gb_linked[-1][1]
            # в массиве j-шек ищем совпадающую с j_prev
            next_el_index = col_matrix_gb.index(j_prev)
            # переносим эл-т с найденным индексом из matrix_gb_copy в matrix_gb_linked
            matrix_gb_linked.append(matrix_gb_copy.pop(next_el_index))
            # удаляем из массивов i-шек и j-шек перенесенный эл-т, чтобы было соотв. matrix_gb
            row_matrix_gb.pop(next_el_index)
            col_matrix_gb.pop(next_el_index)

        matrix_gb_linked.append(matrix_gb_copy.pop())

        matrix_gb_minus = []
        matrix_gb_minus_x_values = []
        for k in range(1, len(matrix_gb_linked), 2):
            i, j = matrix_gb_linked[k]
            matrix_gb_minus.append( (i, j) )
            matrix_gb_minus_x_values.append(matrix_x[i][j])

        theta = min(matrix_gb_minus_x_values)
        matrix_x_old = np.copy(matrix_x)

        for k in range(len(matrix_gb_linked)):
            (i, j) = matrix_gb_linked[k]
            if k % 2 == 0:           # with +
                matrix_x[i][j] += theta
            else:                   # with -
                matrix_x[i][j] -= theta

        log(f'\tОбновленный В = {[(_b[0]+1, _b[1]+1) for _b in array_b]}')
        log(f'\tЭлементы цикла в Gb (произвольный порядок): {[(el[0] + 1, el[1] + 1) for el in matrix_gb]}')
        log(f'\tПравильный Gb (Gb[1] - добавленный индекс): {[(el[0] + 1, el[1] + 1) for el in matrix_gb_linked]}')
        log(f'\tЭлементы Gb с -: {[(el[0]+1, el[1]+1) for el in matrix_gb_minus]}')
        log(f'\tСоответствующие им значения из Х: {matrix_gb_minus_x_values}')
        log(f'\ttheta = min() = {theta}')
        log(f'\tХ =\n{matrix_x_old}')
        log(f'\tОбновленный X = \n{matrix_x}')

        for (i, j) in matrix_gb:
            if matrix_x[i][j] == 0:
                array_b.remove((i, j))
                break

        log(f'\tУдаляем из B {(i + 1, j + 1)}')
        log(f'\tОбновленный В = {[(_b[0] + 1, _b[1] + 1) for _b in array_b]}')
