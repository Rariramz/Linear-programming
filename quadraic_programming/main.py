import itertools as itrt
import numpy as np


def quadratic_programming_task(vector_c, matrix_d, matrix_a, vector_x, vector_jb, vector_jb_star, logger=print):
    def log(message):
        if logger:
            logger(message)

    m = len(matrix_a)
    n = len(matrix_a[0])

    iteration = 1
    while True:
        log(f'\n~~~~~~~~~~~~ ИТЕРАЦИЯ {iteration} ~~~~~~~~~~~~')

        log(f'\nШАГ 1: Находим векторы c(x), u(x), delta(x):')
        log(f'\tx = {vector_x}')
        vector_cx = vector_c + vector_x @ matrix_d
        log(f'\tc(x) = c + D * x = {vector_cx}')
        vector_cbx = np.array([vector_cx[vector_jb[i] - 1] for i in range(len(vector_jb))])
        log(f'\tcb(x) = {vector_cbx}')
        matrix_ab = np.array([(matrix_a[:, vector_jb[i] - 1]) for i in range(len(vector_jb))])
        log(f'\tAb =\n{matrix_ab}')
        matrix_ab_inverse = np.linalg.inv(matrix_ab)
        log(f'\tAb_inverse =\n{matrix_ab_inverse}')
        vector_ux = -vector_cbx @ matrix_ab_inverse
        log(f'\tu(x) = {vector_ux}')
        vector_deltax = vector_ux @ matrix_a + vector_cx
        log(f'\tdelta(x) = {vector_deltax}')

        log(f'\nШАГ 2: Проверяем условие оптимальности текущего правильного опорного плана. ')
        log(f'Если delta(x) >= 0, то текущий правильный опорный план является оптимальным.')
        if min(vector_deltax) >= 0:
            log(f'\nОтвет: X = {vector_x}, B = {vector_jb} - оптимальный план.')
            return vector_x, vector_jb
        log('\tТекущий правильный опорный план не является оптимальным.')

        log(f'\nШАГ 3: Выбираем отрицательную компоненту плана delta(x) и индекс выбранной компоненты заносим в j0:')
        j0 = list(vector_deltax).index(min(vector_deltax))
        log(f'\tj0 = {j0}')

        log(f"\nШАГ 4: По множеству Jb* и j0 найдём вектор l'.")
        log(f"l' делится на два класса. К первому - компоненты с индексами из расширенной опоры ограничений Jb*, ")
        log(f"ко второму - все оставшиеся компоненты.")
        vector_l1 = np.zeros(n)
        vector_l1[j0] = 1
        log(f'\tJb* = {vector_jb_star}')
        l_adv = np.delete(vector_l1, vector_jb_star - 1, axis=0)
        log(f'\tlb* = {l_adv}')

        log(f"\nЧтобы найти lb* составляем матрицу matrix_h = [[D*, Ab'*], [Ab*, 0]] и обращаем её:")
        matrix_d_star_indx = list(itrt.combinations_with_replacement(vector_jb_star - 1, 2))
        len_jb_star = len(vector_jb_star)
        matrix_d_star = np.zeros([len_jb_star, len_jb_star])

        k = 0
        for i in range(len_jb_star):
            for j in range(i, len_jb_star):
                if matrix_d_star_indx[k][0] != matrix_d_star_indx[k][1]:
                    matrix_d_star[j][i] = matrix_d[matrix_d_star_indx[k][1]][matrix_d_star_indx[k][0]]
                matrix_d_star[i][j] = matrix_d[matrix_d_star_indx[k][0]][matrix_d_star_indx[k][1]]
                k += 1
        log(f'\tD* =\n{matrix_d_star}')

        matrix_ab_star = np.array([(matrix_a[:, vector_jb_star[i] - 1]) for i in range(len(vector_jb_star))])
        log(f'\tAb* =\n{matrix_ab_star}')
        matrix_ab1_star = np.transpose(matrix_ab_star)
        log(f"\tAb'* =\n{matrix_ab1_star}")

        matrix1 = np.row_stack((matrix_d_star, matrix_ab1_star))
        matrix2 = np.row_stack((matrix_ab_star, np.zeros([len(matrix_ab_star[0]), len(matrix_ab1_star)])))
        matrix_h = np.column_stack((matrix1, matrix2))
        log(f'\tH =\n{matrix_h}')
        matrix_h_inverse = np.linalg.inv(matrix_h)
        log(f'\tH_inverse =\n{matrix_h_inverse}')

        log(f"\nСтроим вектор b*. Он состоит из двух частей. Сперва идут элементы столбца матрицы D с индексом j0,\n"
            f"стоящие в строках с индексами из Jb*. Далее идут элементы j0-го столбца матрицы A.")
        b_up = np.array([matrix_d[:, j0][vector_jb_star[i] - 1] for i in range(len(vector_jb_star))])
        log(f'\tb_up = {b_up}')
        b_down = matrix_a[:, j0]
        log(f'\tb_down = {b_down}')
        b_star = np.concatenate((b_up, b_down))
        log(f'\tb* = {b_star}')

        log(f"\nНаходим вектор x по следующей формуле x = -matrix_h(-1) * b*:")
        x_hb = -matrix_h_inverse @ b_star
        log(f'\tx_hb = {x_hb}')

        log('\nВектор lb* - необходимое первые n компонент вектора x')
        vector_lb_star = np.array([x_hb[i] for i in range(len(vector_jb_star))])
        log(f'\tlb* = {vector_lb_star}')

        log("\nВ итоге получаем вектор l':")
        vector_l1 = np.concatenate((vector_lb_star, l_adv))
        log(f"\tl' = {vector_l1}")

        log(f'\nШАГ 5: Для каждого индекса j из Jb* найдем величину theta(j), а также вычислим величину theta(j0):')
        log(f'\tl = {vector_l1}')
        log(f'\tD =\n{matrix_d}')
        delta = vector_l1 @ matrix_d @ vector_l1[:, np.newaxis]
        log(f'\tdelta = {delta}')

        theta = np.full(len(vector_jb_star), np.inf)
        theta_j0 = np.inf
        if delta > 0:
            theta_j0 = abs(vector_deltax[j0]) / delta
        log(f'\ttheta(j0) = {theta_j0}')

        for i in range(len(theta)):
            if vector_l1[i] < 0:
                theta[i] = -vector_x[i] / vector_l1[i]
        theta = np.append(theta, theta_j0)
        log(f'\ttheta = {theta}')

        theta0 = min(theta)
        log(f'\ttheta0 = {theta0}')

        if theta0 == np.inf or theta0 > 1e+16:
            log('\nОтвет: целевая функция задачи не ограничена снизу на множестве допустимых планов.')
            return None

        j_star = j0
        if theta0 != theta_j0:
            j_star = vector_jb_star[list(theta).index(theta0)] - 1
        log(f'\tj* = {j_star}')

        log('\nШАГ 6: Обновим допустимый план x = x + theta0 * l:')
        vector_x = vector_x + theta0 * vector_l1
        log(f'\ttheta0 * l = {theta0 * vector_l1}')
        log(f'\tJ = {vector_jb}, Jb* = {vector_jb_star}')

        last_condition = True
        if j_star == j0:
            vector_jb_star = np.append(vector_jb_star, j_star + 1)
            last_condition = False
        elif j_star + 1 in vector_jb_star and j_star + 1 not in vector_jb:
            vector_jb_star = np.delete(vector_jb_star, np.where(vector_jb_star == j_star + 1))
            last_condition = False
        elif j_star + 1 in vector_jb:
            s = list(vector_jb).index(j_star + 1)
            J_adv_tmp = set(vector_jb_star) - set(vector_jb)
            J_adv_tmp = list(J_adv_tmp)
            log(f'\tJ_adv_tmp = {J_adv_tmp}')
            for i in range(len(J_adv_tmp)):
                j_plus = J_adv_tmp[i]
                vector_tmp = matrix_ab_inverse @ matrix_a[:, j_plus - 1]
                print(vector_tmp)
                if vector_tmp[s] != 0:
                    vector_jb = np.where(vector_jb == j_star + 1, j_plus, vector_jb)
                    vector_jb_star = np.delete(vector_jb_star, np.where(vector_jb_star == j_plus))
                    last_condition = False
                    break

        if last_condition:
            vector_jb = np.where(vector_jb == j_star + 1, j0 + 1, vector_jb)
            vector_jb_star = np.where(vector_jb_star == j_star + 1, j0 + 1, vector_jb_star)
        iteration += 1


if __name__ == "__main__":
    c = np.array([-8, -6, -4, -6])
    matrix_d = np.array([[2, 1, 1, 0],
                         [1, 1, 0, 0],
                         [1, 0, 1, 0],
                         [0, 0, 0, 0]])
    matrix_a = np.array([[1, 0, 2, 1],
                         [0, 1, -1, 2]])
    x = np.array([2, 3, 0, 0])
    J = np.array([1, 2])
    vector_jb_star = np.array([1, 2])
    quadratic_programming_task(c, matrix_d, matrix_a, x, J, vector_jb_star)
