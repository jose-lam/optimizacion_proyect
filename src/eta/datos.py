"""M5 - Construccion del dataset de ETA (division temporal por dias, sin fuga de informacion)."""
import numpy as np


def construir_dataset(g, sim, cfg, stride=4):
    tc = cfg["trafico"]
    spd = 1440 // tc["paso_min"]
    Tv, H = tc["ventana_T"], tc["horizonte_min"] // tc["paso_min"]
    T = sim.T
    taus = np.arange(Tv - 1, T - H, stride)
    taus = taus[(taus + H) // spd == taus // spd]                 # el objetivo queda en el mismo dia
    v0 = g.vel_kmh[None, :]
    hist = sim.v_obs[taus[:, None] + np.arange(-Tv + 1, 1)[None, :]] / v0[None]     # (n, Tv, m)
    hist = hist.transpose(0, 2, 1)                                                     # (n, m, Tv)
    tipo1h = np.eye(4)[g.tipo]
    est = np.concatenate([np.log(g.long_m)[:, None], tipo1h, g.carriles[:, None].astype(float), (g.vel_kmh / 100)[:, None]], 1)
    hora = (taus % spd) * tc["paso_min"] / 60.0
    tiempo = np.stack([np.sin(2 * np.pi * hora / 24), np.cos(2 * np.pi * hora / 24)], 1)
    n, m = len(taus), g.m
    X = np.concatenate([hist, np.broadcast_to(est[None], (n, m, est.shape[1])),
                        np.broadcast_to(tiempo[:, None, :], (n, m, 2))], 2).astype(np.float32)
    y = (sim.t_inst[taus + H] / g.t0_s[None, :]).astype(np.float32)                   # razon t/t0
    dia = taus // spd
    nd = tc["dias"]
    tr, va = int(0.70 * nd), int(0.85 * nd)
    idx = {"train": np.where(dia < tr)[0], "val": np.where((dia >= tr) & (dia < va))[0], "test": np.where(dia >= va)[0]}
    mu = X[idx["train"]].reshape(-1, X.shape[2]).mean(0); sd = X[idx["train"]].reshape(-1, X.shape[2]).std(0) + 1e-6
    return dict(X=(X - mu) / sd, X_raw=X, y=y, taus=taus, dia=dia, idx=idx, mu=mu, sd=sd, H=H, spd=spd, t0=g.t0_s)
