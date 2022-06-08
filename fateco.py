"""Основной скрипт для запуска различных численных расчетов."""
import numpy as np
from numpy.linalg import norm
import scipy.signal as sps
import sys
from time import perf_counter as tpc


from func import Func1
from func import Func2
from func import Func3
from plot import plot_dims
from plot import plot_dims_short
from plot import plot_dims_add
from plot import plot_dims_add_short
from plot import plot_dims_conv
from plot import plot_dims_mul
from plot import plot_dims_mul_short
from tensor_train import tt_add
from tensor_train import tt_build
from tensor_train import tt_conv
from tensor_train import tt_full
from tensor_train import tt_get
from tensor_train import tt_mul
from tensor_train import tt_round
from utils import get_tensor_size


def run_demo(d=5, n=10, m=1.E+3, e=1.E-12):
    """Сжатие многомерных функций.

    Демонстрационный расчет с малопараметрическим приближением в рамках
    разложения тензорного поезда (TT-разложения) модельных многомерных
    аналитических функций. Результат будет отображен непосредственно в консоли.

    """
    for FuncClass in [Func1, Func2, Func3]:
        func = FuncClass(d)

        A = func.build_tensor(n)

        t = tpc()
        B = tt_build(A, e)
        t = tpc() - t

        I = func.index_random(n, m)
        X = func.index_to_point(I, n)
        Y_real = func.compute(X)
        Y_calc = tt_get(B, I)
        eps = norm(Y_calc - Y_real) / norm(Y_real)

        print('\n' + '-' * 30)
        print(f'Функция                      : {func.name}')
        print(f'Размер исходного тензора     : {get_tensor_size(A):-8.2e}')
        print(f'Размер TT-тензора            : {get_tensor_size(B):-8.2e}')
        print(f'Относительная ошибка         : {eps:-8.2e}')
        print(f'Время построения (сек)       : {t:-8.5f}')

    print('\n\n')


def run_demo_add(d=5, n=20, e=1.E-5):
    """Сложение многомерных функций.

    Демонстрационный расчет сложения двух TT-тензоров с использованием
    модельных многомерных аналитических функций. Результат будет отображен
    непосредственно в консоли.

    """
    FuncClasses = [
        [Func1, Func1],
        [Func2, Func2],
        [Func3, Func3],
        [Func1, Func2],
        [Func1, Func3],
        [Func2, Func3],
    ]

    for FuncClass1, FuncClass2 in FuncClasses:
        func1 = FuncClass1(d)
        func2 = FuncClass2(d)

        A_np = func1.build_tensor(n)
        B_np = func2.build_tensor(n)

        A_tt = tt_build(A_np, e)
        B_tt = tt_build(B_np, e)

        t_np = tpc()
        C_np = A_np + B_np
        t_np = tpc() - t_np

        t_tt = tpc()
        C_tt = tt_add(A_tt, B_tt)
        t_tt = tpc() - t_tt

        C_tt_full = tt_full(C_tt)

        eps = norm(C_tt_full - C_np) / norm(C_np)

        print('\n' + '-' * 30)
        print(f'Функции (сложение)     : {func1.name} + {func2.name}')
        print(f'Время (полный тензор)  : {t_np:-8.5f}')
        print(f'Время (TT-add)         : {t_tt:-8.5f}')
        print(f'Ошибка результата      : {eps:-8.2e}')

    print('\n\n')


def run_demo_conv(d=5, n=20, e=1.E-5):
    """Свертка многомерных данных.

    Демонстрационный расчет по вычислению многомерной свертки с использованием
    TT-разложения. Результат будет отображен непосредственно в консоли.

    """
    FuncClasses = [
        [Func1, Func1],
        [Func2, Func2],
        [Func3, Func3],
        [Func1, Func2],
        [Func1, Func3],
        [Func2, Func3],
    ]

    for FuncClass1, FuncClass2 in FuncClasses:
        func1 = FuncClass1(d)
        func2 = FuncClass2(d)

        A_np = func1.build_tensor(n)
        B_np = func2.build_tensor(n)

        A_tt = tt_build(A_np, e)
        B_tt = tt_build(B_np, e)

        t_np = tpc()
        C_np = sps.fftconvolve(A_np, B_np, mode='same')
        t_np = tpc() - t_np

        t_tt = tpc()
        C_tt = tt_conv(A_tt, B_tt)
        t_tt = tpc() - t_tt

        print('\n' + '-' * 30)
        print(f'Функции (свертка )     : {func1.name} -conv- {func2.name}')
        print(f'Время (полный тензор)  : {t_np:-8.5f}')
        print(f'Время (TT-conv)        : {t_tt:-8.5f}')
        print(f'Размер полного тензора : {get_tensor_size(C_np):-8.2e}')
        print(f'Размер TT-тензора      : {get_tensor_size(C_tt):-8.2e}')

    print('\n\n')


def run_demo_mul(d=5, n=20, e=1.E-5):
    """Умножение многомерных функций.

    Демонстрационный расчет умножения двух TT-тензоров с использованием
    модельных многомерных аналитических функций. Результат будет отображен
    непосредственно в консоли.

    """
    FuncClasses = [
        [Func1, Func1],
        [Func2, Func2],
        [Func3, Func3],
        [Func1, Func2],
        [Func1, Func3],
        [Func2, Func3],
    ]

    for FuncClass1, FuncClass2 in FuncClasses:
        func1 = FuncClass1(d)
        func2 = FuncClass2(d)

        A_np = func1.build_tensor(n)
        B_np = func2.build_tensor(n)

        A_tt = tt_build(A_np, e)
        B_tt = tt_build(B_np, e)

        t_np = tpc()
        C_np = A_np * B_np
        t_np = tpc() - t_np

        t_tt = tpc()
        C_tt = tt_mul(A_tt, B_tt)
        t_tt = tpc() - t_tt

        C_tt_full = tt_full(C_tt)

        eps = norm(C_tt_full - C_np) / norm(C_np)

        print('\n' + '-' * 30)
        print(f'Функции (умножение)    : {func1.name} * {func2.name}')
        print(f'Время (полный тензор)  : {t_np:-8.5f}')
        print(f'Время (TT-mul)         : {t_tt:-8.5f}')
        print(f'Ошибка результата      : {eps:-8.2e}')

    print('\n\n')


def run_dims(d_list=[3, 4, 5, 6, 7, 8], n=10, m=1.E+3, e=1.E-12):
    """Зависимость степени сжатия от размерности функции.

    Исследование зависимости степени сжатия и точности аппроксимации модельных
    многомерных аналитических функций от размерности задачи с использованием
    малопараметрического разложения тензорного поезда (TT-разложения).
    Результат будет представлен в виде ряда графиков в папке `results`.

    """
    result = {}

    for d in d_list:
        for FuncClass in [Func1, Func2, Func3]:
            func = FuncClass(d)
            if not func.name in result:
                result[func.name] = []

            A = func.build_tensor(n)

            t = tpc()
            B = tt_build(A, e)
            t = tpc() - t

            I = func.index_random(n, m)
            X = func.index_to_point(I, n)
            Y_real = func.compute(X)
            Y_calc = tt_get(B, I)

            compr = get_tensor_size(A) /get_tensor_size(B)
            eps = norm(Y_calc - Y_real)/ norm(Y_real)

            text = func.name + ' '*(20-len(func.name)) + '| '
            text += f'размерность: {d:-2d} | '
            text += f'сжатие: {compr:-7.1e} | '
            text += f'ошибка: {eps:-7.1e} | '
            text += f'время: {t:-7.1e} | '
            print(text)

            result[func.name].append({'dimension': d,
                'compression': compr, 'error': eps, 'time': t})

    print('\n\n')
    plot_dims(result, d_list, f'./results/dims.png')


def run_dims_add(d_list=[3, 4, 5, 6, 7], n=10, e=1.E-6):
    """Эффективность вычисления суммы тензоров в TT-формате.

    Исследование эффективности вычисления суммы тензоров в TT-формате для
    модельных многомерных аналитических функций в зависимости от размерности
    задачи. Результат будет представлен в виде ряда графиков в папке `results`.

    """
    FuncClasses = [
        [Func1, Func1],
        [Func2, Func2],
        [Func3, Func3],
        [Func1, Func2],
        [Func1, Func3],
        [Func2, Func3],
    ]

    result = {}

    for d in d_list:
        for FuncClass1, FuncClass2 in FuncClasses:
            func1 = FuncClass1(d)
            func2 = FuncClass2(d)
            name = func1.name + '+' + func2.name
            name_short = func1.name[0] + '+' + func2.name[0]

            if not name_short in result:
                result[name_short] = []

            A_np = func1.build_tensor(n)
            B_np = func2.build_tensor(n)

            A_tt = tt_build(A_np, e)
            B_tt = tt_build(B_np, e)

            t_np = tpc()
            C_np = A_np + B_np
            t_np = tpc() - t_np

            t_tt = tpc()
            C_tt = tt_add(A_tt, B_tt)
            t_tt = tpc() - t_tt

            C_tt_full = tt_full(C_tt)

            C_tt_round = tt_round(C_tt, e)

            acceleration = t_np / t_tt
            compr = get_tensor_size(C_np) /get_tensor_size(C_tt_round)
            eps = norm(C_tt_full - C_np) / norm(C_np)

            text = name + ' '*(30-len(name)) + '| '
            text += f'размерность: {d:-2d} | '
            text += f'сжатие: {compr:-7.1e} | '
            text += f'ошибка: {eps:-7.1e} | '
            text += f'ускорение: {acceleration:-7.1e} | '
            print(text)

            result[name_short].append({'dimension': d,
                'compression': compr, 'error': eps,
                'acceleration': acceleration})

    print('\n\n')
    plot_dims_add(result, d_list, f'./results/dims_add.png')


def run_dims_conv(d_list=[3, 4, 5, 6], n=10, e=1.E-6):
    """Зависимость эффективности вычисления свертки от размерности.

    Исследование эффективности вычисления многомерных сверток при использовании
    TT-разложения. Результат будет представлен в виде ряда графиков в папке
    `results`.

    """
    FuncClasses = [
        [Func1, Func1],
        [Func2, Func2],
        [Func3, Func3],
        [Func1, Func2],
        [Func1, Func3],
        [Func2, Func3],
    ]

    result = {}

    for d in d_list:
        for FuncClass1, FuncClass2 in FuncClasses:
            func1 = FuncClass1(d)
            func2 = FuncClass2(d)
            name = func1.name + ' -conv- ' + func2.name
            name_short = func1.name[0] + ' -conv- ' + func2.name[0]

            if not name_short in result:
                result[name_short] = []

            A_np = func1.build_tensor(n)
            B_np = func2.build_tensor(n)

            A_tt = tt_build(A_np, e)
            B_tt = tt_build(B_np, e)

            t_np = tpc()
            C_np = sps.fftconvolve(A_np, B_np, mode='same')
            t_np = tpc() - t_np

            t_tt = tpc()
            C_tt = tt_conv(A_tt, B_tt)
            t_tt = tpc() - t_tt

            acceleration = t_np / t_tt
            compr = get_tensor_size(C_np) /get_tensor_size(C_tt)

            text = name + ' '*(30-len(name)) + '| '
            text += f'размерность: {d:-2d} | '
            text += f'сжатие: {compr:-7.1e} | '
            text += f'ускорение: {acceleration:-7.1e} | '
            print(text)

            result[name_short].append({'dimension': d,
                'compression': compr, 'acceleration': acceleration})

    print('\n\n')
    plot_dims_conv(result, d_list, f'./results/dims_conv.png')


def run_dims_mul(d_list=[3, 4, 5, 6, 7], n=10, e=1.E-6):
    """Эффективность вычисления произведения тензоров в TT-формате.

    Исследование эффективности вычисления произведения тензоров в TT-формате для
    модельных многомерных аналитических функций в зависимости от размерности
    задачи. Результат будет представлен в виде ряда графиков в папке `results`.

    """
    FuncClasses = [
        [Func1, Func1],
        [Func2, Func2],
        [Func3, Func3],
        [Func1, Func2],
        [Func1, Func3],
        [Func2, Func3],
    ]

    result = {}

    for d in d_list:
        for FuncClass1, FuncClass2 in FuncClasses:
            func1 = FuncClass1(d)
            func2 = FuncClass2(d)
            name = func1.name + '*' + func2.name
            name_short = func1.name[0] + '*' + func2.name[0]

            if not name_short in result:
                result[name_short] = []

            A_np = func1.build_tensor(n)
            B_np = func2.build_tensor(n)

            A_tt = tt_build(A_np, e)
            B_tt = tt_build(B_np, e)

            t_np = tpc()
            C_np = A_np * B_np
            t_np = tpc() - t_np

            t_tt = tpc()
            C_tt = tt_mul(A_tt, B_tt)
            t_tt = tpc() - t_tt

            C_tt_full = tt_full(C_tt)

            C_tt_round = tt_round(C_tt, e)

            acceleration = t_np / t_tt
            compr = get_tensor_size(C_np) /get_tensor_size(C_tt_round)
            eps = norm(C_tt_full - C_np) / norm(C_np)

            text = name + ' '*(30-len(name)) + '| '
            text += f'размерность: {d:-2d} | '
            text += f'сжатие: {compr:-7.1e} | '
            text += f'ошибка: {eps:-7.1e} | '
            text += f'ускорение: {acceleration:-7.1e} | '
            print(text)

            result[name_short].append({'dimension': d,
                'compression': compr, 'error': eps,
                'acceleration': acceleration})

    print('\n\n')
    plot_dims_mul(result, d_list, f'./results/dims_mul.png')


if __name__ == '__main__':
    np.random.seed(42)

    mode = sys.argv[1] if len(sys.argv) > 1 else 'demo'

    if mode == 'demo':
        run_demo()
    elif mode == 'demo_add':
        run_demo_add()
    elif mode == 'demo_conv':
        run_demo_conv()
    elif mode == 'demo_mul':
        run_demo_mul()
    elif mode == 'dims':
        run_dims()
    elif mode == 'dims_add':
        run_dims_add()
    elif mode == 'dims_conv':
        run_dims_conv()
    elif mode == 'dims_mul':
        run_dims_mul()
    else:
        raise ValueError('Неизвестный тип расчета')
