"""M5 - Perdida de Huber, funcion de influencia y gradiente analitico (solo NumPy). Parte 1, ec. 14-16."""
import numpy as np


def rho(e, delta):
    e = np.asarray(e, float)
    return np.where(np.abs(e) <= delta, 0.5 * e ** 2, delta * (np.abs(e) - 0.5 * delta))


def psi(e, delta):
    e = np.asarray(e, float)
    return np.where(np.abs(e) <= delta, e, delta * np.sign(e))


def perdida_lineal(theta, X, y, delta, lam):
    return float(np.mean(rho(y - X @ theta, delta)) + lam * np.sum(theta ** 2))


def grad_perdida_lineal(theta, X, y, delta, lam):
    """grad L = -(1/N) X^T psi(y - X theta) + 2 lam theta."""
    N = X.shape[0]
    return -(X.T @ psi(y - X @ theta, delta)) / N + 2.0 * lam * theta


def paso_descenso(theta, X, y, delta, lam, eta):
    return theta - eta * grad_perdida_lineal(theta, X, y, delta, lam)
