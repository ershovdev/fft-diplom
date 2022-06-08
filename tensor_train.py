"""Базовый код для построения разложения тензорного поезда."""
import numpy as np


from utils import orthogonalize
from utils import svd


def tt_add(A, B):
    """Построение суммы двух TT-тензоров A и B одинаковых размеров.

    Функция для двух переданных d-мерных тензоров A и B, представленных в
    TT-формате (в форме списков из TT-ядер, являющихся трехмерными массивами),
    возвращает тензор C в TT-формате (в форме списка из TT-ядер), являющийся
    приближенной суммой переданных тензоров: C = A + B.

    """
    n = np.array([G.shape[1] for G in A], dtype=int)
    r1 = np.array([1] + [G.shape[2] for G in A], dtype=int)
    r2 = np.array([1] + [G.shape[2] for G in B], dtype=int)

    C = []
    for i, (G1, G2, k) in enumerate(zip(A, B, n)):
        if i == 0:
            G = np.concatenate([G1, G2], axis=2)
        elif i == len(n) - 1:
            G = np.concatenate([G1, G2], axis=0)
        else:
            r1_l, r1_r = r1[i:i+2]
            r2_l, r2_r = r2[i:i+2]
            Z1 = np.zeros([r1_l, k, r2_r])
            Z2 = np.zeros([r2_l, k, r1_r])
            L1 = np.concatenate([G1, Z1], axis=2)
            L2 = np.concatenate([Z2, G2], axis=2)
            G = np.concatenate([L1, L2], axis=0)
        C.append(G)

    return C


def tt_build(A, e=1E-10, r=1.E+12):
    """Построение сжатого TT-тензора B по заданному полному тензору A.

    Функция для переданного d-мерного тензора A возвращает сжатый тензор в
    TT-формате в форме списка из d TT-ядер, являющихся трехмерными массивами.

    Параметр e соответствует желаемой точности приближения, параметр r - это
    максимальный ранг разложения (может быть указано большое число, если
    ограничение на ранг отсутствует; в этом случае размер сжатого тензора
    будет определяться параметром точности e).

    """
    n = A.shape   # Размеры мод исходного тензора
    Z = A.copy()  # Копия исходного тензора для последующих трансформаций
    B = []        # Тензор в TT-формате в форме списка TT-ядер
    q = 1         # Текущий размер "правого" TT-ранга

    for k in n[:-1]:
        Z = Z.reshape(q * k, -1)
        U, s, V = np.linalg.svd(Z, full_matrices=False, hermitian=False)

        r = max(1, min(int(r), sum(s > e)))
        S = np.diag(np.sqrt(s[:r]))

        G = U[:, :r] @ S
        Z = S @ V[:r, :]

        G = G.reshape(q, k, -1)
        q = G.shape[-1]

        B.append(G)

    B.append(Z.reshape(q, n[-1], 1))

    return B


def tt_conv(A, B, e=1E-10):
    """Вычисление многомерной свертки для двух переданных массивов данных.

    Функция для переданных d-мерных тензоров A и B, осуществляет вычисление их свертки с использованием TT-разложения и возвращает результат в TT-формате.

    """
    # Сохраняем форму исходных тензоров:
    N = [G.shape[1] for G in A]

    # Вычисление FFT для каждого из TT-ядер исходных тензоров:
    A_fft = [_fft_core(G) for G in A]
    B_fft = [_fft_core(G) for G in B]

    # Перемножение тензоров:
    C_fft = tt_mul(A_fft, B_fft)

    # Округление результата:
    C_fft = tt_round(C_fft, e)

    # Вычисление обратного FFT для каждого из TT-ядер результата:
    C = [_ifft_core(G, n) for G, n in zip(C_fft, N)]

    return C


def tt_get(B, I):
    """Вычисление набора элементов с индексами I для тензора в TT-формате."""
    return np.array([tt_get_one(B, i) for i in I])


def tt_get_one(B, i):
    """Вычисление одного элемента с индексом i для тензора в TT-формате."""
    i = np.asanyarray(i, dtype=int)
    y = B[0][0, i[0], :]
    for j in range(1, len(B)):
        y = np.einsum('q,qp->p', y, B[j][:, i[j], :])
    return y[0]


def tt_full(A):
    """Перевод переданного тензора A из TT-формата в "обычный" полный формат.

    Функция для переданного d-мерного тензора A, представленного в TT-формате
    (в форме списков из TT-ядер, являющихся трехмерными массивами), возвращает
    тензор B в полном (numpy) формате.

    """
    Z = A[0].copy()
    for i in range(1, len(A)):
        Z = np.tensordot(Z, A[i], 1)
    return Z[0, ..., 0]


def tt_mul(A, B):
    """Построение произведения двух TT-тензоров A и B одинаковых размеров.

    Функция для двух переданных d-мерных тензоров A и B, представленных в
    TT-формате (в форме списков из TT-ядер, являющихся трехмерными массивами),
    возвращает тензор C в TT-формате (в форме списка из TT-ядер), являющийся
    приближенным поэлементным произведением переданных тензоров: C = A * B.

    """
    C = []
    for G1, G2 in zip(A, B):
        G = G1[:, None, :, :, None] * G2[None, :, :, None, :]
        G = G.reshape([G1.shape[0]*G2.shape[0], -1, G1.shape[-1]*G2.shape[-1]])
        C.append(G)

    return C


def tt_round(A, e, r=1.E+12):
    """Сжатие переданного TT-тензора A.

    Функция для переданного d-мерного тензора A, представленного в TT-формате
    (в форме списков из TT-ядер, являющихся трехмерными массивами), возвращает
    тензор B в TT-формате (в форме списка из TT-ядер), приближенно совпадающий
    с исходным тензором, но имеющий меньший TT-ранг (меньшее число параметров).

    """
    d = len(A)
    B = orthogonalize(A)
    e = e / np.sqrt(d-1) * np.linalg.norm(B[-1])
    for k in range(d-1, 0, -1):
        r1, n, r2 = B[k].shape
        G = np.reshape(B[k], (r1, n * r2), order='F')
        U, V = svd(G, e, r)
        B[k] = np.reshape(V, (-1, n, r2), order='F')
        B[k-1] = np.einsum('ijk,km', B[k-1], U, optimize=True)
    return B


def _fft_core(G):
    r, m, q = G.shape
    G = G.copy()
    G = np.swapaxes(G, 0, 1)
    G = G.reshape((m, r * q))
    G = np.vstack([G, G[m-2 : 0 : -1, :]])
    G = np.fft.fft(G, axis=0).real
    G = G[:m, :] / (m - 1)
    G[0, :] /= 2.
    G[m-1, :] /= 2.
    G = G.reshape((m, r, q))
    G = np.swapaxes(G, 0, 1)
    return G


def _ifft_core(G, n):
    r, m, q = G.shape
    G = G.copy()
    G = np.swapaxes(G, 0, 1)
    G = G.reshape((m, r * q))
    G = np.fft.ifft(G, axis=0).real
    G = G[:n, :]
    G = G.reshape((m, r, q))
    G = np.swapaxes(G, 0, 1)
    return G
