"""M6 - Prediccion -> pesos -> ruta, y simulador de viaje con recalculo dinamico (umbral rho)."""
from __future__ import annotations
import time
import numpy as np
from busqueda import astar, dijkstra_td
from costo import pesos_arista
from grafo import heuristica_perezosa


def construir_pesos(g, predictor, tau, pesos_pref, escalas):
    """t_hat(tau) -> acotamiento inferior t_hat >= d_e/vmax (garantiza admisibilidad de h) -> arreglo aditivo de M4."""
    t_hat = np.maximum(np.asarray(predictor(tau), float), g.long_m / (g.vmax_kmh / 3.6))
    return pesos_arista(g, pesos_pref, t_hat, escalas)


def ruta_optima(g, s, t, pesos, pesos_pref, escalas):
    """A* con h_costo = (w_t/s_t) * d_hav/vmax: admisible y consistente porque los demas terminos del costo son >= 0."""
    h0 = heuristica_perezosa(g, t, g.vmax_kmh)
    k = pesos_pref.get("w_t", 0.0) / escalas[0]
    return astar(g, s, t, pesos, lambda n: k * h0(n))


def simular_viaje(g, s, t, tau0, sim, predictor, pesos_pref, escalas, dt_recalculo, umbral_rho, oraculo=False, recalcular=True):
    """Devuelve dict con tiempo total (s), cambios de ruta, numero y tiempo medio de recalculos.
    umbral_rho es una fraccion (0.05 = 5 %). oraculo=True: ruta optima dependiente del tiempo con velocidades reales."""
    if oraculo:
        r = dijkstra_td(g, s, t, tau0, sim.tiempo_real)
        return dict(tiempo_total=r.costo, cambios=0, recalculos=0, t_recalculo_s=0.0, ruta=r.ruta)
    pesos = construir_pesos(g, predictor, tau0, pesos_pref, escalas)
    ruta = ruta_optima(g, s, t, pesos, pesos_pref, escalas).ruta
    tau, i, cambios, proximo, nrec, tcomp = tau0, 0, 0, tau0 + dt_recalculo, 0, 0.0
    while ruta[i] != t:
        e = g.idx_arista(ruta[i], ruta[i + 1])
        tau += sim.tiempo_real(e, tau)
        i += 1
        if ruta[i] == t or not recalcular or tau < proximo:
            continue
        proximo = tau + dt_recalculo
        c0 = time.perf_counter()
        pesos = construir_pesos(g, predictor, tau, pesos_pref, escalas)
        restante = ruta[i:]
        costo_viejo = sum(pesos[g.idx_arista(a, b)] for a, b in zip(restante[:-1], restante[1:]))
        nueva = ruta_optima(g, ruta[i], t, pesos, pesos_pref, escalas)
        tcomp += time.perf_counter() - c0; nrec += 1
        if costo_viejo > 0 and (costo_viejo - nueva.costo) / costo_viejo > umbral_rho and nueva.ruta != restante:
            ruta = ruta[:i] + nueva.ruta
            cambios += 1
    return dict(tiempo_total=tau - tau0, cambios=cambios, recalculos=nrec, t_recalculo_s=tcomp / max(nrec, 1), ruta=ruta)


def predictor_reactivo(g, sim, sigma, rng):
    """Estimador de tiempos por arista a partir de la velocidad ACTUAL observada (incluye incidentes inyectados) con ruido
    lognormal de desviacion sigma en cada consulta. Es una aproximacion de 'trafico reactivo'; no es la GNN."""
    def f(tau):
        v = sim.vel_vector(tau) * np.exp(sigma * rng.standard_normal(g.m))
        return 3.6 * g.long_m / np.clip(v, 1.0, None)
    return f
