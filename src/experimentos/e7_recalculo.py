"""E7 - Recalculo dinamico: barrido de rho en {0,1,2,5,10} % con incidentes reactivos inyectados; referencias sin recalculo y oraculo."""
import numpy as np, pandas as pd
from experimentos.comun import *
from datos.trafico_sintetico import simular_trafico
from busqueda import dijkstra
from grafo import haversine
from pipeline import simular_viaje, predictor_reactivo, construir_pesos, ruta_optima

RHOS = [0.0, 0.01, 0.02, 0.05, 0.10]


def main(area="S", n_viajes=80, sigma=0.15, dt=30.0):
    cfg = cargar_config(); rng = np.random.default_rng(cfg["semilla"])
    g, _ = grafo_area(area, cfg); sim = simular_trafico(g, cfg)
    pref = {"w_t": 1.0, "w_d": 0.0, "w_p": 0.0, "w_q": 0.0}; esc = np.ones(4)
    spd = 1440 // cfg["trafico"]["paso_min"]; dia0 = int(0.85 * cfg["trafico"]["dias"])
    filas = []; k = 0
    while k < n_viajes:
        s, t = (int(x) for x in rng.integers(0, g.n, 2))
        if haversine(g.lat[s], g.lon[s], g.lat[t], g.lon[t]) < 2000:
            continue
        tau0 = ((dia0 + rng.integers(0, 5)) * spd + int(rng.integers(int(16 * 60 / 5), int(19 * 60 / 5)))) * sim.paso_s    # hora punta de la tarde
        sim.limpiar_incidentes()
        base = simular_viaje(g, s, t, tau0, sim, predictor_reactivo(g, sim, 0.0, rng), pref, esc, dt, 0.0, recalcular=False)
        r0 = base["ruta"]
        if len(r0) < 6:
            continue
        # incidente REACTIVO: aparece en el paso siguiente a la salida (300 s despues) sobre un tramo de la ruta inicial aun no recorrido
        llegada, tt = [], tau0
        for a, b in zip(r0[:-1], r0[1:]):
            llegada.append(tt - tau0); tt += sim.tiempo_real(g.idx_arista(a, b), tt)
        cand = [i for i in range(len(r0) - 1) if llegada[i] >= 330.0]
        if not cand:
            continue
        mid = int(rng.choice(cand)); e = g.idx_arista(r0[mid], r0[mid + 1])
        sim.añadir_incidente(e, tau0 + sim.paso_s, tau0 + 1800, 0.12)
        sin = simular_viaje(g, s, t, tau0, sim, predictor_reactivo(g, sim, sigma, rng), pref, esc, dt, 0.0, recalcular=False)
        orc = simular_viaje(g, s, t, tau0, sim, None, pref, esc, dt, 0.0, oraculo=True)
        fila = dict(viaje=k, s=s, t=t, t_sin_recalculo=sin["tiempo_total"], t_oraculo=orc["tiempo_total"])
        for rho in RHOS:
            r = simular_viaje(g, s, t, tau0, sim, predictor_reactivo(g, sim, sigma, np.random.default_rng(cfg["semilla"] + k)), pref, esc, dt, rho)
            filas.append(dict(**fila, rho_pct=100 * rho, tiempo_total=r["tiempo_total"], cambios=r["cambios"], recalculos=r["recalculos"], t_recalculo_s=r["t_recalculo_s"],
                              ahorro_vs_sin=sin["tiempo_total"] - r["tiempo_total"], brecha_oraculo=r["tiempo_total"] - orc["tiempo_total"]))
        k += 1
    df = pd.DataFrame(filas); guardar(df, "e7_recalculo")
    print(df.groupby("rho_pct")[["tiempo_total", "cambios", "ahorro_vs_sin", "brecha_oraculo", "t_recalculo_s"]].mean())
    print("sin recalculo:", df.drop_duplicates("viaje").t_sin_recalculo.mean(), " oraculo:", df.drop_duplicates("viaje").t_oraculo.mean())


if __name__ == "__main__":
    main()
