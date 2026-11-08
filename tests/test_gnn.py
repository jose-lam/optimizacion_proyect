import torch
from eta.gnn import agregar, GNNETA, CapaMensaje


def test_paso_mensajes_cadena_3_aristas():
    """Ejemplo de la Parte 1: h0=(0.8,0.2,0.2), phi=estado del vecino, promedio, psi=0.5h+0.5m -> (0.425,0.35,0.275)."""
    h = torch.tensor([[0.8], [0.2], [0.2]], dtype=torch.float64)
    dest = torch.tensor([0, 1, 1, 2]); src = torch.tensor([1, 0, 2, 1])
    for _ in range(2):
        h = 0.5 * h + 0.5 * agregar(h[src], dest, 3, "mean")
    assert torch.allclose(h.squeeze(), torch.tensor([0.425, 0.35, 0.275], dtype=torch.float64))


def test_agregaciones_invariantes_a_permutacion():
    m = torch.randn(10, 4); dest = torch.randint(0, 5, (10,)); p = torch.randperm(10)
    for modo in ("sum", "mean", "max"):
        assert torch.allclose(agregar(m, dest, 5, modo), agregar(m[p], dest[p], 5, modo), atol=1e-6)


def test_formas_y_L0():
    pares = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]); x = torch.randn(3, 8)
    for L in (0, 1, 3):
        assert GNNETA(8, 16, L).forward(x, pares).shape == (3,)
    m0 = GNNETA(8, 16, 0); assert torch.allclose(m0(x, pares), m0(x, pares[:, :0]))   # L=0 no usa el grafo


def test_alcance_L_saltos():
    pares = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]); x = torch.zeros(3, 4); x2 = x.clone(); x2[0] += 1
    torch.manual_seed(0); m = GNNETA(4, 8, 1)
    assert torch.allclose(m(x, pares)[2], m(x2, pares)[2], atol=1e-7)          # e3 no ve a e1 con L=1
    torch.manual_seed(0); m2 = GNNETA(4, 8, 2)
    assert not torch.allclose(m2(x, pares)[2], m2(x2, pares)[2], atol=1e-7)    # con L=2 si
