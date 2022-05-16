import numpy as np
from north_west_corner_rule import north_west_corner_rule
from search_optimal_plan import search_optimal_plan


def matrix_transport_problem(vector_a, array_b, matrix_c, logger=print):
    def log(message):
        if logger:
            logger(message)

    log(f'\nВектор поставок:\n\ta = {vector_a}')
    log(f'Вектор спроса:\n\tb = {array_b}')
    log(f'Матрица стоимостей C:\n{matrix_c}')

    log('\n~~~~~ Матричная транспортная задача ~~~~~')

    log('\nКорректировка условия баланса:')
    if sum(vector_a) == sum(array_b):
        log('\tУсловие баланса выполняется :)')
    elif sum(vector_a) > sum(array_b):
        log('\tУсловие баланса не выполняется. Создаем фиктивный пункт потребления:')
        array_b = np.append(array_b, sum(vector_a) - sum(array_b))
        zero_column = np.zeros((len(vector_a), 1))
        matrix_c = np.hstack((matrix_c, zero_column))
        log(f'\ta = {vector_a}\n\tb = {array_b}\n\tC =\n{matrix_c}\n')
    elif sum(vector_a) < sum(array_b):
        log('\tУсловие баланса не выполняется. Создаем фиктивный пункт производства:')
        vector_a = np.append(vector_a, sum(array_b) - sum(vector_a))
        zero_row = np.zeros((1, len(array_b)))
        matrix_c = np.vstack((matrix_c, zero_row))
        log(f'\ta = {vector_a}\n\tb = {array_b}\n\tC =\n{matrix_c}')

    log('\nФАЗА 1. Метод северо-западного угла')
    matrix_x, array_b = north_west_corner_rule(vector_a, array_b)
    log(f'\tB = {list((i + 1, j + 1) for [i, j] in array_b)}')
    log(f'\tНачальное допустимое базисное решение X =\n{matrix_x}')

    log('\nФАЗА 2. Улучшение базисного плана (поиск оптимального)')
    matrix_x, array_b = search_optimal_plan(matrix_x, array_b, matrix_c, logger)
    log(f'\nОтвет: оптимальное базисное решение Х:\n{matrix_x},\nB = {array_b}')
    return matrix_x, array_b


if __name__ == '__main__':
    # Тестовый пример
    a = np.array([100, 300, 300])
    b = np.array([300, 200, 200])
    c = np.array([[8, 4, 1],
                 [8, 4, 3],
                 [9, 7, 5]])

    matrix_transport_problem(a, b, c)

