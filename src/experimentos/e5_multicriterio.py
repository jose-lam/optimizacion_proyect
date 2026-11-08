"""E5 - Costo multicriterio: percentiles vs min-max, barrido de w_p y punto de quiebre analitico vs numerico."""
import numpy as np, pandas as pd
from experimentos.comun import *
from busqueda import dijkstra
from grafo import construir_grafo
from costo import (limites_percentiles, escalas_desde_limites, pesos_arista, acumular_ruta, punto_quiebre_peaje,
                   punto_quiebre_general, cambios_ruta_vs_wp, quiebre_numerico, comparar_normalizaciones, normalizar_ruta)

PESOS_BASE = {"w_t": 1.0, "w_d": 0.0, "w_p": 0.5, "w_q": 0.0}


def ejemplo_parte1():
    """Autopista (16 min, 6 soles) vs via gratuita (28 min); percentiles [10,40] min y [0,12] soles. w_p* = 0.8."""
    u, v = [0, 1, 0, 2], [1, 3, 2, 3]
    t_s = np.array([480.0, 480.0, 840.0, 840.0]); toll = np.array([3.0, 3.0, 0.0, 0.0])
    g = construir_grafo(4, u, v, np.zeros(4), np.zeros(4), np.ones(4), np.ones(4), peaje=toll)
    t_hat = t_s[np.lexsort((np.array(v), np.array(u)))]
    a = np.array([600.0, 0, 0, 0]); b = np.array([2400.0, 1, 12.0, 1]); esc = escalas_desde_limites(a, b)
    z_t1, z_t2, z_p1 = (960 - 600) / 1800, (1680 - 600) / 1800, 6 / 12
    ana = punto_quiebre_peaje(z_t1, z_t2, z_p1, 1.0)
    num = quiebre_numerico(g, 0, 3, {"w_t": 1.0, "w_d": 0, "w_q": 0}, t_hat, esc, list(np.arange(0, 2.01, 0.1)), tol=1e-10)
    return dict(w_analitico=ana, w_numerico=num, dif=abs(ana - num), z_t1=z_t1, z_t2=z_t2, z_p1=z_p1)


def muestrear_rutas(g, t_hat, K, rng, vectores):
    """Protocolo de limites: K pares OD x varios vectores de pesos -> totales X_j(P). Escalas iniciales provisionales."""
    esc0 = np.array([t_hat.sum() / g.m * 40, g.long_m.mean() * 40, max(g.peaje.mean(), 1e-9) * 40, max(g.energia.mean(), 1e-9) * 40])
    X, rutas = [], []
    pares = [(int(a), int(b)) for a, b in rng.integers(0, g.n, (K, 2)) if a != b]
    for s, t in pares:
        for w in vectores:
            r = dijkstra(g, s, t, pesos_arista(g, w, t_hat, esc0))
            X.append(acumular_ruta(g, r.ruta, t_hat))
    return np.array(X), pares


def main(n_pares=100, area="M"):
    cfg = cargar_config(); rng = np.random.default_rng(cfg["semilla"])
    ej = ejemplo_parte1(); guardar(pd.DataFrame([ej]), "e5_ejemplo_parte1")
    assert abs(ej["w_analitico"] - 0.8) < 1e-12 and ej["dif"] < 1e-3, ej
    g, _ = grafo_area(area, cfg)
    t_hat = g.t0_s * np.random.default_rng(cfg["semilla"]).uniform(1.0, 1.8, g.m)
    vect = [{"w_t": 1}, {"w_d": 1}, {"w_t": 1, "w_p": 1}, {"w_t": 1, "w_q": 1}, {"w_t": 1, "w_d": .5, "w_p": .5, "w_q": .5}]
    X, _ = muestrear_rutas(g, t_hat, 150, rng, vect)
    pb, pa = cfg["normalizacion"]["p_bajo"], cfg["normalizacion"]["p_alto"]
    a, b = limites_percentiles(X, pb, pa); esc_p = escalas_desde_limites(a, b)
    esc_m = escalas_desde_limites(X.min(0), X.max(0))
    pd.DataFrame(dict(criterio=["tiempo_s", "distancia_m", "peaje_soles", "energia_kWh"], P5=a, P95=b, min=X.min(0), max=X.max(0))).to_csv(CSV / "e5_limites.csv", index=False)
    # percentiles vs min-max
    pares = [(int(s), int(t)) for s, t in rng.integers(0, g.n, (n_pares * 2, 2)) if s != t][:n_pares]
    comp, quiebres, barrido = [], [], []
    rejilla = list(np.arange(0.0, 3.01, 0.25))
    for k, (s, t) in enumerate(pares):
        rp = dijkstra(g, s, t, pesos_arista(g, PESOS_BASE, t_hat, esc_p)).ruta
        rm = dijkstra(g, s, t, pesos_arista(g, PESOS_BASE, t_hat, esc_m)).ruta
        Xp = acumular_ruta(g, rp, t_hat)
        comp.append(dict(s=s, t=t, misma_ruta=rp == rm, z_t_percentil=float(normalizar_ruta(Xp[0], a[0], b[0])),
                         z_t_minmax=float((Xp[0] - X[:, 0].min()) / (X[:, 0].max() - X[:, 0].min()))))
        pesos_q = {"w_t": 1.0, "w_d": 0.0, "w_q": 0.0}
        cam = cambios_ruta_vs_wp(g, s, t, pesos_q, t_hat, esc_p, rejilla, tol=1e-9)
        for wnum, ra_, rb_ in cam:
            Xa, Xb = acumular_ruta(g, list(ra_), t_hat), acumular_ruta(g, list(rb_), t_hat)
            wan = punto_quiebre_general(Xa, Xb, a, b, pesos_q)
            quiebres.append(dict(s=s, t=t, w_numerico=wnum, w_analitico=wan, dif=abs(wan - wnum)))
        if k < 20:
            for wp in rejilla:
                r = dijkstra(g, s, t, pesos_arista(g, {**pesos_q, "w_p": wp}, t_hat, esc_p)); Xr = acumular_ruta(g, r.ruta, t_hat)
                barrido.append(dict(s=s, t=t, w_p=wp, tiempo_s=Xr[0], dist_m=Xr[1], peaje=Xr[2], n_nodos=len(r.ruta)))
        comp[-1]["n_rutas_distintas"] = len(cam) + 1
    guardar(pd.DataFrame(comp), "e5_percentiles_vs_minmax"); guardar(pd.DataFrame(barrido), "e5_barrido_wp")
    dq = pd.DataFrame(quiebres); guardar(dq, "e5_quiebres")
    guardar(pd.DataFrame([comparar_normalizaciones(25.0, [5, 10, 12, 25, 30, 40, 120])]), "e5_ejemplo_normalizacion")
    print("quiebres verificados:", len(dq), " max |w*-w_num| =", dq.dif.max() if len(dq) else "n/a")
    assert len(dq) == 0 or dq.dif.max() < 1e-3, "E5 FALLA: quiebre analitico != numerico"


if __name__ == "__main__":
    main()
