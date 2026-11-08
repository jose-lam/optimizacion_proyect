"""Pruebas de referencia: reproducen los ejemplos resueltos a mano de la Parte 1."""
import math
import numpy as np
from grafo import construir_grafo, Grafo
from busqueda import dijkstra, astar, bidireccional, dijkstra_td, verificar_fifo
from costo import normalizar_ruta, punto_quiebre_peaje
from eta.huber import grad_perdida_lineal

A, B, C, D, E = range(5)
XY = {A: (0, 0), B: (3, 4), C: (3, -4), D: (-6, 0), E: (6, 0)}
ARISTAS = {(A, B): 10, (A, C): 5, (A, D): 6, (B, E): 5, (C, E): 9, (D, E): 13}


def grafo5():
    u, v, w = [], [], []
    for (a, b), t in ARISTAS.items():
        u += [a, b]; v += [b, a]; w += [t, t]
    g = construir_grafo(5, u, v, [0] * 5, [0] * 5, [0] * len(u), [1] * len(u))
    pesos = np.array([ARISTAS.get((a, b), ARISTAS.get((b, a))) for a, b in zip(g.origen, g.destino)], float)
    return g, pesos


def h_euclid(destino, vmax_kmh=60.0, sobre=None):
    def h(n):
        if sobre and n in sobre:
            return sobre[n]
        return math.dist(XY[n], XY[destino]) / vmax_kmh * 60.0   # minutos
    return h


def test_dijkstra_5_nodos():
    g, w = grafo5()
    r = dijkstra(g, A, E, w)
    assert r.ruta == [A, C, E] and r.costo == 14
    assert r.expandidos == 5 and r.aristas_examinadas == 9


def test_astar_vmax60():
    g, w = grafo5()
    r = astar(g, A, E, w, h_euclid(E, 60))
    assert r.ruta == [A, C, E] and r.costo == 14
    assert r.expandidos == 3 and r.aristas_examinadas == 5


def test_astar_vmax120():
    g, w = grafo5()
    r = astar(g, A, E, w, h_euclid(E, 120))
    assert r.ruta == [A, C, E] and r.costo == 14 and r.expandidos == 5


def test_astar_inadmisible():
    g, w = grafo5()
    r = astar(g, A, E, w, h_euclid(E, 60, sobre={C: 20}))
    assert r.ruta == [A, B, E] and r.costo == 15


def test_consulta_A_D():
    g, w = grafo5()
    rd = dijkstra(g, A, D, w)
    ra = astar(g, A, D, w, h_euclid(D, 60))
    assert rd.costo == ra.costo == 6 and rd.expandidos == 3 and rd.aristas_examinadas == 5
    assert ra.expandidos == 2 and ra.aristas_examinadas == 3


def test_costos_reducidos():
    h = h_euclid(E)
    cp = lambda a, b: ARISTAS.get((a, b), ARISTAS.get((b, a))) + h(b) - h(a)
    assert cp(A, C) == 4 and cp(C, E) == 4 and cp(A, B) == 9 and cp(D, E) == 1
    assert all(cp(a, b) >= 0 and cp(b, a) >= 0 for (a, b) in ARISTAS)


def test_bidireccional_5_nodos():
    g, w = grafo5()
    r = bidireccional(g, A, E, w)
    assert r.ruta == [A, C, E] and r.costo == 14


def test_normalizacion_percentiles():
    assert abs(normalizar_ruta(25, 10, 40) - 0.5) < 1e-12
    assert normalizar_ruta(120, 10, 40) == 1.0


def test_punto_quiebre():
    assert abs(punto_quiebre_peaje(0.2, 0.6, 0.5, w_t=1.0) - 0.8) < 1e-12
    c1 = lambda wp: 0.2 + 0.5 * wp
    assert c1(0.5) < 0.6 and abs(c1(0.8) - 0.6) < 1e-12 and c1(1.2) > 0.6


def test_paso_gradiente_huber():
    X = np.array([[2.0], [3.0], [1.0]]); y = np.array([10.0, 12.0, 20.0]); th = np.array([4.0])
    g = grad_perdida_lineal(th, X, y, delta=5.0, lam=0.01)
    assert abs(g[0] + 2.92) < 1e-12
    assert abs((th - 0.1 * g)[0] - 4.292) < 1e-12


def test_fifo_ejemplo_parte1():
    te_viola = lambda e, tau: 10.0 if tau <= 0 else 4.0   # te(0)=10, te(5)=4
    assert not verificar_fifo(te_viola, 0, [0.0, 5.0])
    te_ok = lambda e, tau: 10.0 if tau <= 0 else 6.0      # te(5)=6
    assert verificar_fifo(te_ok, 0, [0.0, 5.0])


def test_dijkstra_td_constante_igual_estatico():
    g, w = grafo5()
    r = dijkstra_td(g, A, E, 100.0, lambda e, tau: w[e])
    assert r.ruta == [A, C, E] and abs(r.costo - 14) < 1e-12
