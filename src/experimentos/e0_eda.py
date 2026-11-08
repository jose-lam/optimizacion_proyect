"""E0 - Caracterizacion de la red por area (tabla de la seccion 12) + validaciones de M1."""
import numpy as np, pandas as pd
from experimentos.comun import cargar_config, grafo_area, guardar, meta
from datos.descargar_red import validar_grafo


def main():
    cfg = cargar_config(); meta()
    filas, hist = [], []
    for area in ("S", "M", "L"):
        g, st = grafo_area(area, cfg)
        val = validar_grafo(g)
        assert all(val.values()), (area, val)
        filas.append(dict(area=area, nodos=g.n, aristas=g.m, grado_medio=g.m / g.n, long_total_km=g.long_m.sum() / 1000,
                          vmax_kmh=g.vmax_kmh, pct_vel_imputada=st["pct_vel_imputada"], paralelas_descartadas=st["paralelas_descartadas"],
                          nodos_descartados=st["nodos_descartados"], pct_peaje=100 * (g.peaje > 0).mean(), **{f"ok_{k}": v for k, v in val.items()}))
        hist.append(pd.DataFrame(dict(area=area, long_m=g.long_m, vel_kmh=g.vel_kmh, tipo=g.tipo)))
        grado = np.bincount(g.origen, minlength=g.n)
        hist.append(pd.DataFrame(dict(area=area, grado_salida=grado)))
    guardar(pd.DataFrame(filas), "e0_tamanos")
    guardar(pd.concat(hist[::2], ignore_index=True), "e0_aristas")
    guardar(pd.concat(hist[1::2], ignore_index=True), "e0_grados")


if __name__ == "__main__":
    main()
