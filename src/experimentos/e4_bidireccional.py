"""E4 - Bidireccional frente a Dijkstra y A* (mismos pares). Jerarquias de contraccion: NO implementadas."""
import numpy as np, pandas as pd
from experimentos.comun import *
from busqueda import dijkstra, astar, bidireccional
from grafo import heuristica_perezosa


def main(areas=("S", "M", "L"), n_max=150):
    cfg = cargar_config(); rows = []
    for area in areas:
        g, _ = grafo_area(area, cfg)
        w = pesos_congestionados(g, cfg["semilla"])
        rng = np.random.default_rng(cfg["semilla"])
        pares = pares_estratificados(g, min(n_max, cfg["consultas"]["n_por_estrato"][area]), rng)
        for s, t, estrato, _ in pares:
            res = {}
            for nombre, f in (("dijkstra", lambda: dijkstra(g, s, t, w)),
                              ("astar", lambda: astar(g, s, t, w, heuristica_perezosa(g, t, g.vmax_kmh))),
                              ("bidireccional", lambda: bidireccional(g, s, t, w))):
                r, tm = mediana_tiempo(f, 3); res[nombre] = (r, tm)
            ref = res["dijkstra"][0].costo
            for nombre, (r, tm) in res.items():
                rows.append(dict(area=area, estrato=estrato, algoritmo=nombre, expandidos=r.expandidos, aristas=r.aristas_examinadas,
                                 tiempo_s=tm, costo=r.costo, mismo_costo=abs(r.costo - ref) < 1e-9 * max(ref, 1)))
    df = pd.DataFrame(rows); guardar(df, "e4_bidireccional")
    assert df.mismo_costo.all(), "E4: costos distintos"
    print(df.groupby(["area", "algoritmo"])[["expandidos", "tiempo_s"]].mean())


if __name__ == "__main__":
    main()
