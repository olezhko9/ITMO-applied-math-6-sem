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


def pirson(arr, a, k):
    """
    Вычисляет хи-квадрат для заданной выборке
    :param arr: массив чисел - выборка
    :param a: уровень значимости
    :param k: количество интервалов разбиения
    :return: True, если совокупность подчиняется нормальному закону, иначе False.
    Также возвращает вычисленное значение хи-квадрат.
    """

    _min = np.min(arr)
    _max = np.max(arr)
    h = (_max - _min) / k

    x = _min
    k_intervals = {}
    while x < _max:
        xi = np.round(x + h, 3)
        k_intervals[(x, xi)] = {'ni': 0}
        x = xi

    for x in arr:
        for interval in k_intervals:
            if interval[0] <= x <= interval[1]:
                k_intervals[interval]['ni'] += 1
                break

    mean = np.mean(arr)
    std = stdev(arr)
    chisquare = 0
    chi_crit = st.chi2.isf(q=a, df=k - 2 - 1)

    for interval in k_intervals:
        x1 = (interval[0] - mean) / std
        x2 = (interval[1] - mean) / std
        k_intervals[interval]["pi"] = laplass(x2) - laplass(x1)
        chisquare += ((k_intervals[interval]['ni'] - len(arr) * k_intervals[interval]["pi"]) ** 2) / \
                     (len(arr) * k_intervals[interval]["pi"])

    if chisquare > chi_crit:
        return False, chisquare, chi_crit
    else:
        return True, chisquare, chi_crit


def confidence_interval(arr, y):
    """Доверительные интервалы"""
    # для математического ожидания
    t = 1.96  # аргумент функции Лапласса, при котором ее значение равно y / 2 = 0,475
    e = t * sem(arr)
    mean_interval = (np.mean(arr) - e, np.mean(arr) + e)

    # для среднего квадратичного отклонения
    X2 = st.chi2.isf(q=(1-y)/2, df=len(arr) - 1)  # Для количества степеней свободы k = 99 и p = (1-0.95)/2 = 0.025
    tH = (len(arr) - 1) * np.var(arr, ddof=1) / X2

    X2 = st.chi2.isf(q=(1+y)/2, df=len(arr) - 1)  # Для количества степеней свободы k = 99 и p = (1+0.95)/2 = 0.975
    tB = (len(arr) - 1) * np.var(arr, ddof=1) / X2
    std_interval = (np.sqrt(tH), np.sqrt(tB))

    return mean_interval, std_interval


if __name__ == "__main__":
    stat = np.fromfile("sample.txt", dtype=int, sep=' ')
    print(stat)

    # Гистограмма
    sns.distplot(stat, hist=True, bins=10)
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

    print("Пирсон:", pirson(stat, a=0.025, k=10))

    intervals = confidence_interval(stat, 0.95)
    print("Доверительные интервалы для мат. ожидания: {} и среднего квадратичного отклонения: {}:".format(intervals[0], intervals[1]))
