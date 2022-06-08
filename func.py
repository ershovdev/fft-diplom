"""Классы с демонстрационными многомерными функциями для тестирования."""
import numpy as np


class Func:
    """Базовый класс для скалярной функции многомерного аргумента.

    Конструктору передается размерность функции d и область ее определения lim,
    представляющая список из двух чисел (минимальное и максимальное значение
    входного аргумента функции по каждой размерности). Также опционально может
    быть передано отображаемое имя функции name.

    Конкретные функции должны наследовать данный класс и определять метод
    compute, который для переданного набора многомерных аргументов функции
    (в формате двумерного массива samples x dimensions) возвращает список
    соответствующих значений функции (длины samples).

    """
    def __init__(self, d, lim, name='Function'):
        self.d = d
        self.lim = lim
        self.name = name

    def build_tensor(self, n=10):
        """Построение дискретизации функции на равномерной сетке с n-узлами.

        Для d-мерной функции строится тензор (многомерный массив) ее значений
        на равномерной сетке (от минимального значения функции до максимального)
        c n-узлами сетки по каждой размерности (моде).

        """
        # Построение списка всех возможных индексов тензора:
        I = self.index_all(n)

        # Перевод индексов в аргументы функции:
        X = self.index_to_point(I, n)

        # Вычисление значений функции в полном наборе точек:
        Y = self.compute(X)

        # Преобразование списка значений в многомерный тензор:
        Y = Y.reshape([n]*self.d, order='F')

        return Y

    def index_all(self, n):
        """Построение списка всех возможных индексов тензора."""
        I = [np.arange(n).reshape(1, -1) for _ in range(self.d)]
        I = np.meshgrid(*I, indexing='ij')
        I = np.array(I, dtype=int).reshape((self.d, -1), order='F').T

        return I

    def index_random(self, n, m):
        """Построение набора из m случайных индексов тензора."""
        I = np.vstack([np.random.choice(n, int(m)) for _ in range(self.d)]).T
        return I

    def index_to_point(self, I, n):
        """Перевод переданного набора индексов в аргументы функции.

        Предполагается, что переданный набор индексов I соответствует
        равномерной сетке с n узлами по каждой размерности. Функция возвращает
        соответствующий набор входных аргументов функции, отвечающих переданным
        узлам сетки.

        """
        m = I.shape[0]

        a = np.ones(self.d) * self.lim[0]
        b = np.ones(self.d) * self.lim[1]
        n = np.ones(self.d, dtype=int) * n

        a = np.repeat(a.reshape((1, -1)), m, axis=0)
        b = np.repeat(b.reshape((1, -1)), m, axis=0)
        n = np.repeat(n.reshape((1, -1)), m, axis=0)

        X = I / (n - 1) * (b - a) + a

        return X

    def compute(self, X):
        """Вычисление значений функции для переданного набора аргументов."""
        raise NotImplementedError()

    def point_random(self, n, m):
        """Построение набора из m случайных аргументов функции на сетке."""
        I = self.index_random(n, m)
        X = self.index_to_point(I, n)

        return X


class Func1(Func):
    """Многомерная функция Exponential для тестирования.

    Простая функция вида -1 * exp(-x^2/2).

    """
    def __init__(self, d):
        super().__init__(d, [-1., +1.], 'Exponential')

    def compute(self, X):
        return -np.exp(-0.5 * np.sum(X**2, axis=1))


class Func2(Func):
    """Многомерная функция Dixon для тестирования.

    См. https://www.sfu.ca/~ssurjano/dixonpr.html

    """
    def __init__(self, d):
        super().__init__(d, [-10., +10.], 'Dixon')

    def compute(self, X):
        y1 = (X[:, 0] - 1)**2
        y2 = np.arange(2, self.d+1) * (X[:, 1:]**2 - X[:, :-1])**2
        y2 = np.sum(y2, axis=1)
        return y1 + y2


class Func3(Func):
    """Многомерная функция Ackley для тестирования.

    См. https://www.sfu.ca/~ssurjano/ackley.html

    """
    def __init__(self, d):
        super().__init__(d, [-32.768, +32.768], 'Ackley')

        self.par_a = 20.
        self.par_b = 0.2
        self.par_c = 2.*np.pi

    def compute(self, X):
        y1 = np.sqrt(np.sum(X**2, axis=1) / self.d)
        y1 = - self.par_a * np.exp(-self.par_b * y1)
        y2 = np.sum(np.cos(self.par_c * X), axis=1)
        y2 = - np.exp(y2 / self.d)
        y3 = self.par_a + np.exp(1.)
        return y1 + y2 + y3
