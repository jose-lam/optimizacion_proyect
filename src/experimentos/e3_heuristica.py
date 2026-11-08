"""E3 - Calidad de la heuristica: barrido de vmax (0.7x a 2.0x) y h=0; degradacion de optimalidad."""
import numpy as np, pandas as pd
from experimentos.comun import *
from busqueda import dijkstra, astar
from grafo import heuristica_perezosa

FACTORES = [0.7, 0.85, 1.0, 1.25, 1.5, 2.0]


def main(areas=("S", "M", "L"), n_max=150):
    cfg = cargar_config(); rows = []
    for area in areas:
        g, _ = grafo_area(area, cfg)
        w = pesos_congestionados(g, cfg["semilla"])
        rng = np.random.default_rng(cfg["semilla"])
        pares = pares_estratificados(g, min(n_max, cfg["consultas"]["n_por_estrato"][area]), rng)
        for s, t, estrato, _ in pares:
            rd = dijkstra(g, s, t, w)
            rows.append(dict(area=area, estrato=estrato, heuristica="h=0", factor=0.0, admisible=True, expandidos=rd.expandidos,
                             costo=rd.costo, costo_opt=rd.costo, optima=True, exceso_rel=0.0))
            for f in FACTORES:
                ra = astar(g, s, t, w, heuristica_perezosa(g, t, g.vmax_kmh * f))   # vmax*f: f<1 sobreestima h
                rows.append(dict(area=area, estrato=estrato, heuristica=f"{f}*vmax_real(h=d/(vmax*{f}))", factor=f, admisible=f >= 1.0,
                                 expandidos=ra.expandidos, costo=ra.costo, costo_opt=rd.costo,
                                 optima=abs(ra.costo - rd.costo) <= 1e-9 * rd.costo, exceso_rel=(ra.costo - rd.costo) / rd.costo))
    df = pd.DataFrame(rows); guardar(df, "e3_heuristica")
    adm = df[df.admisible]
    assert adm.optima.all(), "E3: heuristica admisible produjo ruta no optima"
    print(df.groupby(["area", "factor"]).agg(exp=("expandidos", "mean"), pct_no_opt=("optima", lambda x: 100 * (1 - x.mean())), exceso=("exceso_rel", "mean")))


if __name__ == "__main__":
    main()
