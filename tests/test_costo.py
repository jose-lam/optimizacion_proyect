import numpy as np
from costo import pesos_arista, limites_percentiles, acumular_ruta, escalas_desde_limites, punto_quiebre_general, comparar_normalizaciones
from experimentos.comun import cargar_config, grafo_area
from busqueda import dijkstra

g, _ = grafo_area("S", cargar_config())


def test_aditividad_pesos_arista():
    t = g.t0_s * 1.3; esc = np.array([600., 3000., 2., 1.]); p = {"w_t": 1, "w_d": .3, "w_p": .5, "w_q": .2}
    c = pesos_arista(g, p, t, esc); r = dijkstra(g, 0, 100, c).ruta
    X = acumular_ruta(g, r, t)
    esperado = p["w_t"] * X[0] / esc[0] + p["w_d"] * X[1] / esc[1] + p["w_p"] * X[2] / esc[2] + p["w_q"] * X[3] / esc[3]
    assert abs(sum(c[g.idx_arista(a, b)] for a, b in zip(r[:-1], r[1:])) - esperado) < 1e-9


def test_percentiles_robustos():
    h = np.r_[np.linspace(10, 40, 98), 120.0, 125.0]
    r = comparar_normalizaciones(25.0, h)
    assert r["z_percentil"] > 0.4 and r["z_minmax"] < 0.2     # los atipicos comprimen min-max pero no percentiles


def test_quiebre_general_reduce_a_formula():
    a, b = np.array([10., 0, 0, 0]), np.array([40., 1, 12., 1])
    Xa, Xb = np.array([16., 0, 6, 0]), np.array([28., 0, 0, 0])
    assert abs(punto_quiebre_general(Xa, Xb, a, b, {"w_t": 1.0}) - 0.8) < 1e-12
