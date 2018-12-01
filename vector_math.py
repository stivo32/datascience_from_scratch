def vector_substract(v, w):
    return [v_i - w_i for v_i, w_i in zip(v, w)]


def scalar_multiple(c, v):
    return [c * v_i for v_i in v]
