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


def limit_dist_by_computing(transition_matrix, stationary_dist, eps=1e-7):
    m_stationary_dist = stationary_dist ** 2
    stdev = vector_stdev(stationary_dist, m_stationary_dist)
    m = 0
    while stdev > eps:
        m_stationary_dist = stationary_dist @ transition_matrix
        m += 1
        # print(stationary_dist, m_stationary_dist)
        stdev = vector_stdev(stationary_dist, m_stationary_dist)
        # print("Отклонение", stdev)
        stationary_dist = m_stationary_dist
    return stationary_dist, m


def vector_stdev(va, vb):
    return np.sqrt(sum((b - a)**2 for b, a in zip(vb, va)))




if __name__ == '__main__':
    n = 8
    Markov_matrix = get_random_transition_matrix(n)
    print("Матрица переходов\n", Markov_matrix)

    start_dist_1 = get_random_norm_vector(n)

    print("Вектор начального состояния\n", start_dist_1)
    print("Предельная матрица\n")
    print(limit_dist_by_computing(Markov_matrix, start_dist_1))

    start_dist_2 = get_random_norm_vector(n)
    print("Вектор начального состояния\n", start_dist_2)
    print("Предельная матрица\n")
    print(limit_dist_by_computing(Markov_matrix, start_dist_2))

    print("Матрица в степени\n", np.linalg.matrix_power(Markov_matrix, n))


    # A = (Markov_matrix.transpose() - np.eye(8))
    # B = np.zeros((8, 1))
    # print(np.linalg.inv(A) @ B)
