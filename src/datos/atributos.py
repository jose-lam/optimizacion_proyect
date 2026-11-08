"""M1 - Atributos por arista: velocidad (imputada), t0, peaje y energia (valores ASUMIDOS, configurables)."""
import numpy as np

VEL_POR_TIPO = {0: 30.0, 1: 40.0, 2: 60.0, 3: 80.0}   # residencial, secundaria, arterial, autopista


def imputar_atributos(vel_kmh, tipo):
    """Imputa velocidades faltantes (NaN) con la mediana por tipo de via. Devuelve (vel, porcentaje_imputado)."""
    vel = np.asarray(vel_kmh, float).copy()
    falta = ~np.isfinite(vel)
    for t in np.unique(tipo):
        m = tipo == t
        ref = np.nanmedian(vel[m]) if np.isfinite(vel[m]).any() else VEL_POR_TIPO.get(int(t), 30.0)
        vel[m & falta] = ref
    return vel, 100.0 * falta.mean()


def tiempo_flujo_libre(long_m, vel_kmh):
    return 3.6 * long_m / vel_kmh


def peaje_por_arista(tipo, cfg):
    """Soles. SUPUESTO: tarifa fija por arista de autopista (cfg['peajes_soles']['autopista_por_arista'])."""
    return np.where(tipo == 3, cfg["peajes_soles"]["autopista_por_arista"], 0.0)


def energia_por_arista(long_m, vel_kmh, pend_pos, cfg):
    """q_e = alfa d + beta d g+ + gamma d phi(v), d en km. phi(v)=(v/60)^2. SUPUESTO declarado."""
    e = cfg["energia"]
    d = long_m / 1000.0
    return e["alfa"] * d + e["beta"] * d * pend_pos + e["gamma"] * d * (vel_kmh / 60.0) ** 2
