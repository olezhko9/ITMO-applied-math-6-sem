import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st


def quartil(arr):
    """квартили"""
    q1, q2, q3 = 0, 0, 0
    arr = sorted(arr)
    q2 = np.median(arr)
    mid = int(np.floor(len(arr) / 2))
    if len(arr) % 2 == 0:
        q1 = np.median(arr[:mid + 1])
        q3 = np.median(arr[mid:])
    else:
        q1 = np.median(arr[:mid])
        q3 = np.median(arr[mid + 1:])

    return q1, q2, q3


def stdev(arr):
    """Стандартное отклонение"""
    mean = np.mean(arr)
    return np.sqrt(sum([(x - mean) ** 2 for x in arr]) / (len(arr) - 1))


def sem(arr):
    """Стандартная ошибка"""
    return stdev(arr) / np.sqrt(len(arr))


def kurtosis(arr):
    """эксцесс"""
    return sum((arr - np.mean(arr)) ** 4) / len(arr) / stdev(stat) ** 4 - 3


def assimetria(arr):
    return sum((arr - np.mean(arr)) ** 3) / len(arr) / stdev(stat) ** 3


def laplass(x):
    return st.norm.cdf(x) - 0.5


def pirson(arr, a, k, chi_crit):
    """
    Вычисляет хи-квадрат для заданной выборке
    :param arr: массив чисел - выборка
    :param a: уровень значимости
    :param k: количество интервалов разбиения
    :param chi_crit: хи-квадрат критическое при аргументах (k-2-1, a)
    :return: True, если совокупность подчиняется нормальному закону, иначе False.
    Также возвращает вычисленное значение хи-квадрат.
    """
    _k = k
    _min = np.min(arr)
    _max = np.max(arr)
    h = (_max - _min) / _k

    x = _min
    k_intervals = {}
    while x < _max:
        xi = np.round(x + h, 3)
        k_intervals[(x, xi)] = {'ni': 0}
        x = xi

    for a in arr:
        for interval in k_intervals:
            if interval[0] <= a <= interval[1]:
                k_intervals[interval]['ni'] += 1
                break

    mean = np.mean(arr)
    std = stdev(arr)
    chisquare = 0
    for interval in k_intervals:
        x1 = (interval[0] - mean) / std
        x2 = (interval[1] - mean) / std
        k_intervals[interval]["pi"] = laplass(x2) - laplass(x1)
        chisquare += ((k_intervals[interval]['ni'] - len(arr) * k_intervals[interval]["pi"]) ** 2) / \
                     (len(arr) * k_intervals[interval]["pi"])

    if chisquare > chi_crit:
        return False, chisquare
    else:
        return True, chisquare


def confidence_interval(arr):
    """Доверительные интервалы"""
    # для математического ожидания
    t = 1.96  # аргумент функции Лапласса, при котором ее значение равно 0,95 / 2 = 0,475
    e = t * sem(arr)
    mean_interval = (np.mean(arr) - e, np.mean(arr) + e)

    # для дисперсии
    # Случайная ошибка дисперсии нижней границы
    X2 = 129.5612  # Для количества степеней свободы k = 99 и p = (1-0.95)/2 = 0.025 по таблице распределения χ2
    tH = (len(arr) - 1) * np.var(arr) / X2
    # Случайная ошибка дисперсии верхней границы
    X2 = 74.22193  # Для количества степеней свободы k = 99 и p = (1-0.95)/2 = 0.975 по таблице распределения χ2
    tB = (len(arr) - 1) * np.var(arr) / X2
    var_interval = (np.sqrt(tH), np.sqrt(tB))

    return mean_interval, var_interval


if __name__ == "__main__":
    stat = np.fromfile("sample.txt", dtype=int, sep=' ')
    print(stat)

    # Гистограмма
    sns.distplot(stat, hist=True, kde=True, bins=10)
    plt.show()
    # Эмперический график
    plt.hist(stat, 100, normed=1, histtype='step', cumulative=True, label='Empirical')
    plt.show()

    print("Минимум: {}, максимум: {}".format(stat.min(), stat.max()))

    print("Среднее:", np.mean(stat))

    print("Дисперсия:", np.var(stat, ddof=1))

    print("Стандртная ошибка:", sem(stat))

    print("Мода:", st.mode(stat))

    print("Медиана:", np.median(stat))

    print("Квартили:", quartil(stat), (np.percentile(stat, 25), np.percentile(stat, 50), np.percentile(stat, 75)))

    # ящик с усами
    sns.boxplot(stat)
    plt.show()

    print("Стандартное отклонение:", stdev(stat))

    print("Эксцесс:", kurtosis(stat), st.kurtosis(stat))

    print("Асимметрия:", assimetria(stat), st.skew(stat))

    print("Пирсон:", pirson(stat, a=0.025, k=10, chi_crit=12.833))

    intervals = confidence_interval(stat)
    print("Доверительные интервалы для мат. ожидания: {} и среднего квадратичного отклонения: {}:".format(intervals[0], intervals[1]))
    print(st.t.interval(0.95, len(stat)-1, loc=np.mean(stat), scale=st.sem(stat)))