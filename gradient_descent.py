from random import randint
import math
from datascience_from_scratch.support_funcs import negate, negate_all


# Функция потерь
def square(v):
    return [v_i * v_i for v_i in v]


# Градиент суммы квадратов
def sum_of_squares_gradient(v):
    return [2 * v_i for v_i in v]


# Двигаться с шаговым размером step_size в направлении от v
def step(v, direction, step_size):
    #  direction <- градиент
    return [
        v_i + step_size * direction_i
        for v_i, direction_i
        in zip(v, direction)
    ]


def safe(f):
    def safe_f(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except:
            return float('inf')
    return safe_f


def distance(v, w):
    return math.sqrt(sum([(w_i - v_i) ** 2 for v_i, w_i in zip(v, w)]))


# пакетная минимизация с помощью градиентного спуска
def minimize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    """
    :param target_fn: функция, которую надо минимизировать
    :param gradient_fn: градиент функции
    :param theta_0: начальный вектор параметров
    :param tolerance: точность
    :return: theta, при котором функция минимальная
    """
    step_sizes = [100, 10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]  # вектор шагов, будет выбираться самый эффективный
    theta = theta_0
    target_fn = safe(target_fn)
    value = target_fn(theta)  # значение функции

    while True:
        gradient = gradient_fn(theta)
        next_thetas = [
            step(theta, gradient, -step_size)
            for step_size
            in step_sizes
        ]
        next_theta = min(next_thetas, key=target_fn)
        next_value = target_fn(next_theta)
        if abs(value - next_value) < tolerance:
            return theta
        else:
            theta, value = next_theta, next_value


# для случая, когда нужно максимизировать функцию
def maximize_batch(target_fn, gradient_fn, theta_0, tolerance=0.000001):
    return minimize_batch(
        negate(target_fn),
        negate_all(gradient_fn),
        theta_0,
        tolerance
    )


v = [randint(-10, 10) for _ in range(3)]  # произвольная отправная точка
tolerance = 0.0000001  # константа точности расчета


while True:
    gradient = sum_of_squares_gradient(v)
    next_v = step(v, gradient, -0.01)
    if distance(next_v, v) < tolerance:
        break
    v = next_v

print(v)
