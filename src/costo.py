"""M4 - Funcion costo multicriterio, normalizacion por percentiles y punto de quiebre (Parte 1, ec. 9-11, 17-18)."""
from __future__ import annotations
import numpy as np
from busqueda import dijkstra

CRITERIOS = ("tiempo", "distancia", "peaje", "energia")


def limites_percentiles(X: np.ndarray, p_bajo=5, p_alto=95):
    """X: (n_rutas, n_criterios) -> (a, b) por criterio; a=P5, b=P95."""
    X = np.atleast_2d(np.asarray(X, float))
    return np.percentile(X, p_bajo, axis=0), np.percentile(X, p_alto, axis=0)


def escalas_desde_limites(a, b):
    """s_j = b_j - a_j (si el criterio es casi constante, s_j = 1 para evitar division por 0)."""
    s = np.asarray(b, float) - np.asarray(a, float)
    return np.where(s > 1e-12, s, 1.0)


def pesos_arista(g, pesos: dict, t_hat: np.ndarray, escalas: np.ndarray) -> np.ndarray:
    """c_e = sum_j w_j x_ej / s_j. Aditivo, SIN recorte (el recorte rompe la aditividad)."""
    s = np.asarray(escalas, float)
    return (pesos.get("w_t", 0.0) * np.asarray(t_hat, float) / s[0] + pesos.get("w_d", 0.0) * g.long_m / s[1]
            + pesos.get("w_p", 0.0) * g.peaje / s[2] + pesos.get("w_q", 0.0) * g.energia / s[3])


def normalizar_ruta(X_ruta, a, b, recortar=True):
    """z_j = (X_j - a_j)/(b_j - a_j); recorte a [0,1] solo para evaluar rutas completas."""
    z = (np.asarray(X_ruta, float) - a) / (np.asarray(b, float) - a)
    return np.clip(z, 0.0, 1.0) if recortar else z


def acumular_ruta(g, ruta, t_hat) -> np.ndarray:
    """Totales X_j(P) = (tiempo, distancia, peaje, energia) de una ruta."""
    e = np.array([g.idx_arista(u, v) for u, v in zip(ruta[:-1], ruta[1:])], int)
    return np.array([np.asarray(t_hat)[e].sum(), g.long_m[e].sum(), g.peaje[e].sum(), g.energia[e].sum()])


def comparar_normalizaciones(x_tipica, historial, a_pct=5, b_pct=95):
    """Percentiles vs min-max absolutos para un valor tipico (efecto de atipicos)."""
    a, b = np.percentile(historial, [a_pct, b_pct])
    lo, hi = np.min(historial), np.max(historial)
    return {"z_percentil": float(np.clip((x_tipica - a) / (b - a), 0, 1)), "z_minmax": float((x_tipica - lo) / (hi - lo))}


def punto_quiebre_peaje(z_t1, z_t2, z_p1, w_t=1.0, z_p2=0.0) -> float:
    """w_p* = w_t (z_t2 - z_t1)/(z_p1 - z_p2); con z_p2 = 0 es la ec. (18)."""
    return w_t * (z_t2 - z_t1) / (z_p1 - z_p2)


def punto_quiebre_general(Xa, Xb, a, b, pesos: dict) -> float:
    """w_p* entre dos rutas con totales Xa, Xb y los demas pesos fijos (transformacion afin, sin recorte)."""
    za, zb = (np.asarray(Xa) - a) / (np.asarray(b) - a), (np.asarray(Xb) - a) / (np.asarray(b) - a)
    w = np.array([pesos.get("w_t", 0), pesos.get("w_d", 0), 0.0, pesos.get("w_q", 0)])
    return float(((w * zb).sum() - (w * za).sum()) / (za[2] - zb[2]))


def cambios_ruta_vs_wp(g, s, t, pesos, t_hat, escalas, rejilla, tol=1e-7):
    """Barre w_p; en cada cambio de ruta optima biseca. Devuelve [(w_num, ruta_antes, ruta_despues)]."""
    def ruta(wp):
        return tuple(dijkstra(g, s, t, pesos_arista(g, {**pesos, "w_p": wp}, t_hat, escalas)).ruta)
    out, prev = [], ruta(rejilla[0])
    for i in range(1, len(rejilla)):
        cur = ruta(rejilla[i])
        if cur != prev:
            lo, hi = rejilla[i - 1], rejilla[i]
            while hi - lo > tol:
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if ruta(mid) == prev else (lo, mid)
            out.append((0.5 * (lo + hi), prev, ruta(hi)))
            prev = cur
    return out


def quiebre_numerico(g, s, t, pesos, t_hat, escalas, rejilla, tol=1e-7) -> float:
    c = cambios_ruta_vs_wp(g, s, t, pesos, t_hat, escalas, rejilla, tol)
    return c[0][0] if c else float("nan")


def penalizacion_ruta(g, ruta, umbral_grados=45.0) -> int:
    """Omega(P): numero de giros bruscos (no aditiva en aristas). Solo para evaluar rutas."""
    ang = []
    for u, v in zip(ruta[:-1], ruta[1:]):
        ang.append(np.arctan2(g.lat[v] - g.lat[u], (g.lon[v] - g.lon[u]) * np.cos(g.lat[u])))
    d = np.abs(np.degrees(np.diff(ang)))
    d = np.minimum(d, 360 - d)
    return int((d > umbral_grados).sum())
