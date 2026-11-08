import copy
import numpy as np, networkx as nx
from experimentos.comun import cargar_config, grafo_area, pesos_congestionados, a_networkx
from busqueda import dijkstra, astar, bidireccional, dijkstra_array, dijkstra_td, verificar_fifo
from grafo import heuristica_perezosa, verificar_consistencia, heuristica_tiempo
from datos.descargar_red import validar_grafo

cfg = cargar_config(); g, _ = grafo_area("S", cfg); w = pesos_congestionados(g, 42)


def test_m1_validaciones():
    assert all(validar_grafo(g).values())


def test_consistencia_heuristica():
    assert verificar_consistencia(g, np.array(w), g.vmax_kmh) and verificar_consistencia(g, np.array(w), g.vmax_kmh, destino=3)
    assert not verificar_consistencia(g, g.t0_s / 2, g.vmax_kmh)   # tiempos por debajo de d/vmax rompen la condicion


def test_astar_h0_igual_dijkstra():
    for s, t in [(0, 100), (5, 60), (130, 2)]:
        a, d = astar(g, s, t, w, lambda n: 0.0), dijkstra(g, s, t, w)
        assert a.ruta == d.ruta and a.expandidos == d.expandidos


def test_costos_vs_networkx():
    G = a_networkx(g, w); r = np.random.default_rng(1)
    for _ in range(40):
        s, t = (int(x) for x in r.integers(0, g.n, 2))
        ref = nx.dijkstra_path_length(G, s, t)
        for res in (dijkstra(g, s, t, w), astar(g, s, t, w, heuristica_perezosa(g, t, g.vmax_kmh)), bidireccional(g, s, t, w), dijkstra_array(g, s, t, w)):
            assert abs(res.costo - ref) <= 1e-9 * max(ref, 1)
            assert abs(sum(w[g.idx_arista(a, b)] for a, b in zip(res.ruta[:-1], res.ruta[1:])) - ref) <= 1e-9 * max(ref, 1)


def test_no_modifica_entradas():
    w0, ip0, d0 = list(w), g.indptr.copy(), g.destino.copy()
    dijkstra(g, 0, 50, w); astar(g, 0, 50, w, heuristica_perezosa(g, 50, g.vmax_kmh)); bidireccional(g, 0, 50, w)
    assert w == w0 and (g.indptr == ip0).all() and (g.destino == d0).all()


def test_heuristica_vectorizada_igual_perezosa():
    h = heuristica_tiempo(g, 17, g.vmax_kmh); hp = heuristica_perezosa(g, 17, g.vmax_kmh)
    assert all(abs(h[n] - hp(n)) < 1e-6 for n in range(0, g.n, 7))


def test_instrumentacion():
    r = dijkstra(g, 0, 100, w)
    assert r.expandidos >= len(r.ruta) and r.aristas_examinadas > 0 and r.inserciones >= r.expandidos and r.max_cola >= 1 and r.tiempo_s > 0
