"""Вспомогательные функции."""
import numpy as np
import scipy as sp


def get_tensor_size(A):
    """Возвращает число элементов полного тензора или TT-тензора."""
    if isinstance(A, list):
        # Передан TT-tensor, представленный в виде списка ядер:
        return np.sum([G.size for G in A])
    else:
        # Передан обычный многомерный массив:
        return A.size


def orthogonalize(A, k=None):
    """Производит полную ортогонализация ядер переданного TT-тензора."""
    Y = [G.copy() for G in A]

    d = len(Y)
    k = d - 1 if k is None else k

    R = np.array([[1.]])
    for i in range(k):
        R = orthogonalize_left(Y, i)

    L = np.array([[1.]])
    for i in range(d-1, k, -1):
        L = orthogonalize_right(Y, i)

    return Y


def orthogonalize_left(Y, k):
    """Производит левую ортогонализация ядер переданного TT-тензора."""
    d = len(Y)

    r1, n1, r2 = Y[k].shape
    G1 =np.reshape(Y[k], (r1 * n1, r2), order='F')
    Q, R = np.linalg.qr(G1, mode='reduced')
    Y[k] = np.reshape(Q, (r1, n1, Q.shape[1]), order='F')

    r2, n2, r3 = Y[k+1].shape
    G2 = np.reshape(Y[k+1], (r2, n2 * r3), order='F')
    G2 = R @ G2
    Y[k+1] = np.reshape(G2, (G2.shape[0], n2, r3), order='F')

    return R


def orthogonalize_right(Y, k):
    """Производит правую ортогонализация ядер переданного TT-тензора."""
    d = len(Y)
    r2, n2, r3 = Y[k].shape
    G2 = np.reshape(Y[k], (r2, n2 * r3), order='F')
    L, Q = sp.linalg.rq(G2, mode='economic', check_finite=False)
    Y[k] = np.reshape(Q, (Q.shape[0], n2, r3), order='F')

    r1, n1, r2 = Y[k-1].shape
    G1 = np.reshape(Y[k-1], (r1 * n1, r2), order='F')
    G1 = G1 @ L
    Y[k-1] = np.reshape(G1, (r1, n1, G1.shape[1]), order='F')

    return L


def svd(A, e=1.E-10, r=1.E+12):
    """Строит усеченное SVD для переданной матрицы A."""
    m, n = A.shape
    C = A @ A.T if m <= n else A.T @ A

    if np.linalg.norm(C) < 1.E-12:
        return np.zeros([m, 1]), np.zeros([1, n])

    w, U = np.linalg.eigh(C)

    w[w < 0] = 0.
    w = np.sqrt(w)

    idx = np.argsort(w)[::-1]
    w = w[idx]
    U = U[:, idx]

    s = w**2
    where = np.where(np.cumsum(s[::-1]) <= e**2)[0]
    dlen = 0 if len(where) == 0 else int(1 + where[-1])
    rank = max(1, min(int(r), len(s) - dlen))
    w = w[:rank]
    U = U[:, :rank]

    V = ((1. / w)[:, np.newaxis] * U.T) @ A if m <= n else U.T
    U = U * w if m <= n else A @ U

    return U, V
