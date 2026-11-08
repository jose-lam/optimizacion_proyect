"""E2 - Dijkstra vs A* sobre pares estratificados; E3 y E4 reutilizan el muestreo."""
import tracemalloc
import numpy as np, pandas as pd
from experimentos.comun import *
from busqueda import dijkstra, astar
from grafo import heuristica_perezosa


def main(areas=("S", "M", "L")):
    cfg = cargar_config(); rows = []
    for area in areas:
        g, _ = grafo_area(area, cfg)
        w = pesos_congestionados(g, cfg["semilla"])
        rng = np.random.default_rng(cfg["semilla"])
        pares = pares_estratificados(g, cfg["consultas"]["n_por_estrato"][area], rng)
        for k, (s, t, estrato, dkm) in enumerate(pares):
            rd, td = mediana_tiempo(lambda: dijkstra(g, s, t, w), 3, calentar=1 if k < 5 else 0)
            ra, ta = mediana_tiempo(lambda: astar(g, s, t, w, heuristica_perezosa(g, t, g.vmax_kmh)), 3)
            rows.append(dict(area=area, estrato=estrato, dist_recta_km=dkm, s=s, t=t,
                             costo_dij=rd.costo, costo_ast=ra.costo, mismo_costo=abs(rd.costo - ra.costo) < 1e-9 * max(rd.costo, 1),
                             exp_dij=rd.expandidos, exp_ast=ra.expandidos, ar_dij=rd.aristas_examinadas, ar_ast=ra.aristas_examinadas,
                             ins_dij=rd.inserciones, ins_ast=ra.inserciones, maxq_dij=rd.max_cola, maxq_ast=ra.max_cola,
                             t_dij=td, t_ast=ta))
        # memoria (corrida aparte, no distorsiona los tiempos)
        mem = []
        for s, t, estrato, _ in pares[:: max(1, len(pares) // 30)]:
            pk = []
            for f in (lambda: dijkstra(g, s, t, w), lambda: astar(g, s, t, w, heuristica_perezosa(g, t, g.vmax_kmh))):
                tracemalloc.start(); f(); pk.append(tracemalloc.get_traced_memory()[1] / 1024); tracemalloc.stop()
            mem.append(dict(area=area, estrato=estrato, mem_dij_kb=pk[0], mem_ast_kb=pk[1]))
        guardar(pd.DataFrame(mem), f"e2_memoria_{area}")
    df = pd.DataFrame(rows); guardar(df, "e2_dijkstra_vs_astar")
    assert df.mismo_costo.all(), "E2: A* con h admisible devolvio costo distinto"
    print(df.groupby(["area", "estrato"])[["exp_dij", "exp_ast", "t_dij", "t_ast"]].mean())


if __name__ == "__main__":
    main()
