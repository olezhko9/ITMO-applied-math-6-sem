import numpy as np
import random
random.seed(2019)


def get_random_norm_vector(n):
    a = []
    norm_sum = 1.0
    for i in range(n - 1):
        x = random.triangular(0, norm_sum)
        norm_sum -= x
        a.append(x)
    a.append(norm_sum)
    a = np.array(a)
    # np.random.shuffle(a)
    return a


def get_random_transition_matrix(n):
    g = []
    for i in range(n):
        g.append(get_random_norm_vector(n))
    return np.array(g)

def vector_stdev(va, vb):
    return np.sqrt(sum((b - a)**2 for b, a in zip(vb, va)))


def limit_dist_by_computing(transition_matrix, stationary_dist, eps=1e-5):
    m_stationary_dist = stationary_dist ** 2
    stdev = vector_stdev(stationary_dist, m_stationary_dist)
    m = 0
    while stdev > eps:
        m_stationary_dist = stationary_dist @ transition_matrix
        m += 1
        stdev = vector_stdev(stationary_dist, m_stationary_dist)
        stationary_dist = m_stationary_dist
    return stationary_dist, m


def limit_dist_by_analytic(transition_matrix):
    A = (transition_matrix - np.eye(n)).transpose()
    probability_dist = np.ones((1, n))

    A = np.vstack((A, probability_dist))
    B = np.zeros(n + 1)
    B[-1] = 1

    p = np.linalg.lstsq(A, B, rcond=1)[0]
    return p


if __name__ == '__main__':
    n = 8
    Markov_matrix = get_random_transition_matrix(n)
    print("Матрица переходов\n", Markov_matrix)

    # Численный способ
    start_dist_1 = get_random_norm_vector(n)

    print("\nВектор начального состояния 1\n", start_dist_1)
    print("\nЧисленное распределение 1")
    print(limit_dist_by_computing(Markov_matrix, start_dist_1))

    start_dist_2 = get_random_norm_vector(n)
    print("\nВектор начального состояния 2\n", start_dist_2)
    print("\nЧисленное распределение 2")
    print(limit_dist_by_computing(Markov_matrix, start_dist_2))

    print("\nМатрица в степени m\n", np.linalg.matrix_power(Markov_matrix, n))

    # Аналитический способ
    print("\nАналитическое распределение\n", limit_dist_by_analytic(Markov_matrix))
