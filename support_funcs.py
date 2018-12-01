# отрицание результата на выходе
def negate(f):
    # вернуть функцию, которая для любого входящего x возвращает -f(x)
    return lambda *args, **kwargs: -f(*args, **kwargs)


# отрицание списка результатов на выходе
def negate_all(f):
    # то же самое, когда f возвращает список чисел
    return lambda *args, **kwargs: [-y for y in f(*args, **kwargs)]
