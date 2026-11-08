"""M5 - Lineas base: persistencia, media historica por arista y franja, ridge. (El MLP sin grafo es GNNETA con L=0.)"""
import numpy as np
from sklearn.linear_model import Ridge


def persistencia(datos, g, sim, tau_idx):
    """Ultima observacion: razon t/t0 = v0/v_obs(tau)."""
    return (g.vel_kmh[None, :] / sim.v_obs[datos["taus"][tau_idx]]).astype(np.float32)


def media_historica(datos, g, sim, tau_idx, cfg):
    """Media por (arista, franja horaria) de la razon t/t0 en los dias de entrenamiento."""
    spd = datos["spd"]; nd_tr = int(0.70 * cfg["trafico"]["dias"])
    r = (sim.t_inst[: nd_tr * spd] / g.t0_s[None, :]).reshape(nd_tr, spd, g.m).mean(0)       # (spd, m)
    slot = (datos["taus"][tau_idx] + datos["H"]) % spd
    return r[slot].astype(np.float32)


def ridge(datos, alpha=1.0):
    """Regresion ridge con rezagos y variables estaticas (sin estructura de grafo). Devuelve funcion predecir(idx)."""
    X, y, tr = datos["X"], datos["y"], datos["idx"]["train"]
    mod = Ridge(alpha=alpha).fit(X[tr].reshape(-1, X.shape[2]), y[tr].reshape(-1))
    return lambda idx: mod.predict(X[idx].reshape(-1, X.shape[2])).reshape(len(idx), -1).astype(np.float32)
