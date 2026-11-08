import numpy as np
from eta.huber import grad_perdida_lineal, perdida_lineal, rho, psi


def _datos():
    r = np.random.default_rng(0)
    X = r.normal(size=(50, 4)); y = X @ np.array([1., -2, .5, 3]) + r.normal(size=50) * 2; y[:3] += 40   # atipicos
    return X, y, r.normal(size=4)


def test_gradiente_vs_diferencias_finitas():
    X, y, th = _datos(); an = grad_perdida_lineal(th, X, y, 1.5, 0.01); h = 1e-6
    fd = np.array([(perdida_lineal(th + h * np.eye(4)[i], X, y, 1.5, .01) - perdida_lineal(th - h * np.eye(4)[i], X, y, 1.5, .01)) / (2 * h) for i in range(4)])
    assert np.linalg.norm(an - fd) / np.linalg.norm(fd) < 1e-6


def test_gradiente_vs_autograd():
    import torch
    X, y, th = _datos(); t = torch.tensor(th, requires_grad=True)
    l = torch.nn.functional.huber_loss(torch.tensor(X) @ t, torch.tensor(y), delta=1.5) + 0.01 * (t ** 2).sum()
    l.backward()
    an = grad_perdida_lineal(th, X, y, 1.5, 0.01)
    assert np.linalg.norm(an - t.grad.numpy()) / np.linalg.norm(an) < 1e-6


def test_rho_psi_continuidad():
    d = 2.0
    assert abs(float(rho(d, d)) - 2.0) < 1e-12 and abs(float(rho(d + 1e-9, d)) - 2.0) < 1e-6
    assert float(psi(10.0, d)) == d and float(psi(-10.0, d)) == -d and float(psi(1.0, d)) == 1.0
