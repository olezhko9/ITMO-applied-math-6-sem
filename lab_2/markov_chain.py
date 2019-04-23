import numpy as np
import random
import matplotlib.pyplot as plt
random.seed(2019)


def get_random_norm_vector(n):
    """Возвращает нормированный вектор длины n"""
    a = []
    norm_sum = 1.0
    for i in range(n - 1):
        x = random.triangular(0, norm_sum)
        norm_sum -= x
        a.append(x)
    a.append(norm_sum)
    a = np.array(a)
    np.random.shuffle(a)
    return a


def get_random_transition_matrix(n):
    """Возвращает квадратную матрицу размерности n, состояющую из нормированных векторов"""
    g = []
    for i in range(n):
        g.append(get_random_norm_vector(n))
    return np.array(g)


def vector_stdev(va, vb):
    """Среднеквадратическое отклонение между векторами"""
    return np.sqrt(sum((b - a)**2 for b, a in zip(vb, va)))


def limit_dist_by_computing(transition_matrix, stationary_dist, eps=1e-5):
    """Численное нахождение стационарного состояния"""
    m_stationary_dist = stationary_dist ** 2
    stdev = vector_stdev(stationary_dist, m_stationary_dist)
    std_arr = []
    m = 0
    while stdev > eps:
        m_stationary_dist = stationary_dist @ transition_matrix
        m += 1
        stdev = vector_stdev(stationary_dist, m_stationary_dist)
        std_arr.append(stdev)
        stationary_dist = m_stationary_dist
    return stationary_dist, m, np.array(std_arr)


def limit_dist_by_analytic(transition_matrix):
    """Аналитическое нахождение стационарного состояния"""
    n = len(transition_matrix)
    A = (transition_matrix - np.eye(n)).transpose()
    probability_dist = np.ones((1, n))

    A = np.vstack((A, probability_dist))
    B = np.zeros(n + 1)
    B[-1] = 1

    p = np.linalg.lstsq(A, B, rcond=1)[0]
    return p


if __name__ == '__main__':
    n = 8
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle('Изменение среднеквадратического отклонения', fontsize=16)
    Markov_matrix = get_random_transition_matrix(n)
    print("Матрица переходов\n", np.around(Markov_matrix, 3))

    # Численный способ
    start_dist_1 = get_random_norm_vector(n)

    print("\nВектор начального состояния 1\n", np.around(start_dist_1, 3))
    print("\nЧисленное распределение 1")
    finish_dist_1, m, stdarr = limit_dist_by_computing(Markov_matrix, start_dist_1)
    print(np.around(finish_dist_1, 3), m)
    ax1.plot(np.arange(0, len(stdarr)), stdarr)

    start_dist_2 = get_random_norm_vector(n)
    print("\nВектор начального состояния 2\n", np.around(start_dist_2, 3))
    print("\nЧисленное распределение 2")
    finish_dist_2, m, stdarr = limit_dist_by_computing(Markov_matrix, start_dist_2)
    print(np.around(finish_dist_2, 3), m)
    ax2.plot(np.arange(0, len(stdarr)), stdarr)
    plt.show()

    print("\nМатрица в степени m\n", np.around(np.linalg.matrix_power(Markov_matrix, n), 3))

    # Аналитический способ
    print("\nАналитическое распределение\n", np.around(limit_dist_by_analytic(Markov_matrix), 3))
