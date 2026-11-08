"""E6 - Predictor de ETA: baselines (siempre) + MLP/GNN/ablaciones (requieren torch; si falta, se registra como NO EJECUTADO)."""
import numpy as np, pandas as pd
from experimentos.comun import *
from datos.trafico_sintetico import simular_trafico, lista_vecinos
from eta.datos import construir_dataset
from eta import baselines
from busqueda import dijkstra


def metricas(pred_ratio, y_ratio, t0):
    e = (pred_ratio - y_ratio) * t0[None, :]
    yt = y_ratio * t0[None, :]
    return dict(MAE_s=float(np.abs(e).mean()), RMSE_s=float(np.sqrt((e ** 2).mean())), MAPE_pct=float(100 * (np.abs(e) / yt).mean()))


def eta_ruta(g, sim, datos, pred, test_idx, n_rutas, rng):
    """Error del ETA de ruta: suma de t_hat(tau+H) vs tiempo recorrido real (integrando velocidades) saliendo en tau+H."""
    rows = []
    w = g.t0_s.tolist()
    for _ in range(n_rutas):
        s, t = (int(x) for x in rng.integers(0, g.n, 2))
        if s == t:
            continue
        ruta = dijkstra(g, s, t, w).ruta
        edges = [g.idx_arista(a, b) for a, b in zip(ruta[:-1], ruta[1:])]
        i = int(rng.integers(len(test_idx)))
        tau_d = (datos["taus"][test_idx[i]] + datos["H"]) * sim.paso_s
        real, tt = 0.0, tau_d
        for e in edges:
            d = sim.tiempo_real(e, tt); real += d; tt += d
        rows.append((float(np.sum(g.t0_s[edges] * pred[i, edges])), real))
    p, r = np.array(rows).T
    return dict(ETA_MAE_s=float(np.abs(p - r).mean()), ETA_MAPE_pct=float(100 * (np.abs(p - r) / r).mean()))


def main(area="S"):
    cfg = cargar_config(); rng = np.random.default_rng(cfg["semilla"])
    g, _ = grafo_area(area, cfg)
    sim = simular_trafico(g, cfg); datos = construir_dataset(g, sim, cfg)
    te = datos["idx"]["test"]; y = datos["y"][te]; t0 = datos["t0"]
    # verificacion anti-fuga: ningun dia de test aparece en entrenamiento/validacion
    assert set(datos["dia"][datos["idx"]["train"]]).isdisjoint(datos["dia"][te]) and set(datos["dia"][datos["idx"]["val"]]).isdisjoint(datos["dia"][te])
    preds = {"persistencia": baselines.persistencia(datos, g, sim, te), "media_historica": baselines.media_historica(datos, g, sim, te, cfg),
             "ridge": baselines.ridge(datos)(te)}
    estado = {k: "ejecutado" for k in preds}
    try:
        import torch
        from eta.gnn import GNNETA
        from eta.entrenar import entrenar, predecir, busqueda_hiperparametros
        e = cfg["eta"]; d_in = datos["X"].shape[2]
        pares = torch.as_tensor(lista_vecinos(g))
        mejores = busqueda_hiperparametros(datos, pares, d_in, e["trials"], cfg["semilla"])
        guardar(pd.DataFrame(mejores), "e6_busqueda_hiperparametros")
        b = mejores[0]
        configs = {"MLP_L0": dict(L=0, agg="mean", loss="huber"), "GNN_L1": dict(L=1, agg="mean", loss="huber"),
                   "GNN_L2": dict(L=2, agg="mean", loss="huber"), "GNN_L3": dict(L=3, agg="mean", loss="huber"),
                   "GNN_L2_sum": dict(L=2, agg="sum", loss="huber"), "GNN_L2_max": dict(L=2, agg="max", loss="huber"),
                   "GNN_L2_cuadratica": dict(L=2, agg="mean", loss="cuadratica")}
        curvas = []
        for nombre, c in configs.items():
            torch.manual_seed(cfg["semilla"])
            mod = GNNETA(d_in, b["d"], c["L"], c["agg"])
            h = entrenar(mod, datos, pares, b["eta"], b["lam"], e["delta"], e["epocas"], e["paciencia"], cfg["semilla"], e["lote"], c["loss"])
            curvas += [dict(modelo=nombre, **x) for x in h]
            preds[nombre] = predecir(mod, datos["X"][te], pares); estado[nombre] = "ejecutado"
        guardar(pd.DataFrame(curvas), "e6_curvas")
    except ImportError as ex:
        for nombre in ("MLP_L0", "GNN_L1", "GNN_L2", "GNN_L3", "GNN_L2_sum", "GNN_L2_max", "GNN_L2_cuadratica"):
            estado[nombre] = f"NO EJECUTADO (torch no disponible: {ex})"
    filas = []
    for nombre, estd in estado.items():
        f = dict(modelo=nombre, estado=estd)
        if nombre in preds:
            f.update(metricas(preds[nombre], y, t0)); f.update(eta_ruta(g, sim, datos, preds[nombre], te, 300, np.random.default_rng(cfg["semilla"])))
        filas.append(f)
    guardar(pd.DataFrame(filas), "e6_resultados"); print(pd.DataFrame(filas).to_string())


if __name__ == "__main__":
    main()
