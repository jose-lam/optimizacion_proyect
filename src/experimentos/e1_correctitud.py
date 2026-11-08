"""E1 - Correctitud frente a networkx (diferencia relativa de costo < 1e-9) para todos los algoritmos."""
import numpy as np, pandas as pd, networkx as nx
from experimentos.comun import *
from busqueda import dijkstra, astar, bidireccional
from grafo import heuristica_perezosa, verificar_consistencia


def main(areas=("S", "M", "L")):
    cfg = cargar_config(); rows = []
    for area in areas:
        g, _ = grafo_area(area, cfg)
        w = pesos_congestionados(g, cfg["semilla"])
        assert verificar_consistencia(g, np.array(w), g.vmax_kmh), "heuristica inconsistente"
        G = a_networkx(g, w)
        rng = np.random.default_rng(cfg["semilla"])
        n = cfg["consultas"]["n_correctitud"][area]
        hechos = 0
        while hechos < n:
            s, t = (int(x) for x in rng.integers(0, g.n, 2))
            if s == t:
                continue
            hechos += 1
            ref = nx.dijkstra_path_length(G, s, t)
            for nombre, f in (("dijkstra", lambda: dijkstra(g, s, t, w)),
                              ("astar", lambda: astar(g, s, t, w, heuristica_perezosa(g, t, g.vmax_kmh))),
                              ("bidireccional", lambda: bidireccional(g, s, t, w))):
                r = f()
                rows.append(dict(area=area, algoritmo=nombre, s=s, t=t, costo_ref=ref, costo=r.costo,
                                 dif_rel=abs(r.costo - ref) / max(ref, 1e-12)))
    df = pd.DataFrame(rows); guardar(df, "e1_correctitud")
    print(df.groupby(["area", "algoritmo"]).dif_rel.agg(["count", "max"]))
    assert (df.dif_rel < 1e-9).all(), "E1 FALLA"


if __name__ == "__main__":
    main()
