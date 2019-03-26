import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

# https://www.ibm.com/support/knowledgecenter/ru/SSEP7J_10.1.1/com.ibm.swg.ba.cognos.ug_cr_rptstd.10.1.1.doc/c_id_obj_desc_tables.html

def quartil(arr):
    # квартили
    q1, q2, q3 = 0, 0, 0
    arr = sorted(arr)
    q2 = np.median(arr)
    mid = int(np.floor(len(arr)/2))
    if len(arr) % 2 == 0:
        q1 = np.median(arr[:mid + 1])
        q3 = np.median(arr[mid:])
    else:
        q1 = np.median(arr[:mid])
        q3 = np.median(arr[mid + 1:])

    return q1, q2, q3

def stdev(arr):
    # Стандартное отклонение
    mean = np.mean(arr)
    return np.sqrt(sum([(x - mean)**2 for x in arr]) / (len(arr)-1))

def sem(arr):
    # Стандартная ошибка
    return stdev(arr) / np.sqrt(len(arr))

def kurtosis(arr):
    # эксцесс
    return sum((arr - np.mean(arr))**4) / len(arr) / stdev(stat)**4 - 3

def assimetria(arr):
    return sum((arr - np.mean(arr))**3) / len(arr) / stdev(stat)**3

def pirson(arr):
    k = 10
    _min = np.min(arr)
    _max = np.max(arr)
    h = (_max - _min) / k
    x = _min
    i = 1
    while x < _max:
        xi = np.round(x + h, 3)
        print(i, x, xi)
        i += 1
        x = xi

    return h, _min, _max

def interval(arr):
    # для математического ожидания
    t = 1.96    # аргумент функции Лапласса, при котором ее значение равно 0,95 / 2 = 0,475
    e = t * sem(arr)
    mean_interval = (np.mean(arr) - e, np.mean(arr) + e)

    # для дисперсии
    # Случайная ошибка дисперсии нижней границы
    X2 = 129.5612 # Для количества степеней свободы k = 99 и p = (1-0.95)/2 = 0.025 по таблице распределения χ2
    tH = (len(arr) - 1) * np.var(arr) / X2
    # Случайная ошибка дисперсии верхней границы
    X2 = 74.22193 # Для количества степеней свободы k = 99 и p = (1-0.95)/2 = 0.975 по таблице распределения χ2
    tB = (len(arr) - 1) * np.var(arr) / X2
    var_interval = (tH, tB)

    return mean_interval, var_interval

if __name__ == "__main__":
    stat = np.fromfile("sample.txt", dtype=int, sep=' ')
    print(stat)

    # гистограмма и график функции распределения
    # sns.distplot(stat, hist=True, kde=True, bins=10)
    # plt.show()

    print("Среднее:", np.mean(stat))

    print("Дисперсия:", np.var(stat))

    print("Стандртная ошибка:", sem(stat))

    print("Мода:", st.mode(stat))

    print("Медиана:", np.median(stat))

    # http://math.msu.su/~falin/files/Фалин_Г.И.,Фалин_А.И.(2011-Математика)Квартили_в_описательной_статистике.pdf
    print("Квартили:", quartil(stat), (np.percentile(stat, 25), np.percentile(stat, 50), np.percentile(stat, 75)))

    sns.boxplot(stat)
    plt.show()

    print("Стандартное отклонение:", stdev(stat))

    print("Эксцесс:", kurtosis(stat), st.kurtosis(stat))

    # https: // ru.wikipedia.org / wiki / Коэффициент_асимметрии
    print("Асимметрия:", assimetria(stat), st.skew(stat))

    print("Пирсон:", pirson(stat))

    print("Доверительные интервалы: для мат. ожидания:", interval(stat))
    print(st.t.interval(0.95, len(stat)-1, loc=np.mean(stat), scale=st.sem(stat)))
