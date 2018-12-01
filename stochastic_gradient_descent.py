import random
from datascience_from_scratch.vector_math import vector_substract, scalar_multiple
from datascience_from_scratch.support_funcs import negate, negate_all

"""

Метод стохастического градиентного спуска(stochastic gradient descent) за одну итерацию цикла вычисляет градиент
(и делает шаг) только на одной точке. Он многогратно просмотривает данные, пока не достигнет точки останова.
Во время каждого цикла просмотр данных выполняется в случайном порядке.
"""


# перемешать индексы
def in_random_order(data):
    """
    генератор, который возвращает элементы данных в случайном порядке.
    :param data: набор векторов с данными
    :return: случайный вектор из набора
    """
    indexes = [i for i, _ in enumerate(data)]
    random.shuffle(indexes)
    for i in indexes:
        yield data[i]


# стохастическая минимизация
def minimize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    """
    :param target_fn: функция, которую надо минимизировать
    :param gradient_fn: градиент функции
    :param x:
    :param y:
    :param theta_0: начальный вектор параметров
    :param alpha_0: начальный размер шага
    :return: theta, при котором функция минимальная
    """
    data = zip(x, y)
    theta = theta_0  # первоначальная гипотеза
    alpha = alpha_0  # первоначальный размер шага
    min_theta, min_value = None, float('inf')  # минимум на этот момент
    iterations_with_no_improvement = 0

    # остановиться, если достигли 100 итераций без улучшений
    while iterations_with_no_improvement < 100:
        value = sum(target_fn(x_i, y_i, theta) for x_i, y_i in data)

        if value < min_value:
            # если найден новый минимум, то запомнить его
            # и вернуться к первоначальному размеру шага
            min_theta, min_value = theta, value
            iterations_with_no_improvement = 0
            alpha = alpha_0
        else:
            # если улучшений нет, то пытаемся сжать размер шага
            iterations_with_no_improvement += 1
            alpha *= 0.9

            # и делаем шаг градиент для каждой из точек данных
            for x_i, y_i in in_random_order(data):
                gradient_i = gradient_fn(x_i, y_i, theta)
                theta = vector_substract(theta, scalar_multiple(alpha, gradient_i))
    return min_theta


def maximize_stochastic(target_fn, gradient_fn, x, y, theta_0, alpha_0=0.01):
    return minimize_stochastic(
        negate(target_fn),
        negate_all(gradient_fn),
        x, y, theta_0, alpha_0
        )
