"""M2 - Grafo CSR, Haversine y verificacion de consistencia (Parte 1, ec. 4-6)."""
from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

R_TIERRA_M = 6_371_000.0


def haversine(lat1, lon1, lat2, lon2):
    """Distancia en metros entre puntos (radianes); escalar o vectorizada."""
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_TIERRA_M * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


@dataclass
class Grafo:
    n: int
    m: int
    indptr: np.ndarray      # (n+1,) CSR
    destino: np.ndarray     # (m,)
    lat: np.ndarray         # (n,) radianes
    lon: np.ndarray         # (n,) radianes
    long_m: np.ndarray      # (m,)
    vel_kmh: np.ndarray     # (m,)
    t0_s: np.ndarray        # (m,) flujo libre
    peaje: np.ndarray       # (m,) soles
    energia: np.ndarray     # (m,) proxy kWh
    vmax_kmh: float
    origen: np.ndarray | None = None
    tipo: np.ndarray | None = None
    carriles: np.ndarray | None = None

    def __post_init__(self):
        self._ip = self.indptr.tolist()
        self._dst = self.destino.tolist()
        if self.origen is None:
            self.origen = np.repeat(np.arange(self.n), np.diff(self.indptr))
        self._inv = None
        self._idx = None

    def idx_arista(self, u: int, v: int) -> int:
        if self._idx is None:
            self._idx = {(int(a), int(b)): i for i, (a, b) in enumerate(zip(self.origen, self.destino))}
        return self._idx[(u, v)]

    def inversa(self):
        """CSR del grafo transpuesto: (indptr, origen_de_arista, indice_de_arista_original)."""
        if self._inv is None:
            orden = np.argsort(self.destino, kind="stable")
            cuenta = np.bincount(self.destino, minlength=self.n)
            rip = np.concatenate([[0], np.cumsum(cuenta)])
            self._inv = (rip.tolist(), self.origen[orden].tolist(), orden.tolist())
        return self._inv


def construir_grafo(n, u, v, lat, lon, long_m, vel_kmh, peaje=None, energia=None,
                    tipo=None, carriles=None, vmax_kmh=None) -> Grafo:
    u, v = np.asarray(u, int), np.asarray(v, int)
    orden = np.lexsort((v, u))
    u, v = u[orden], v[orden]
    tomar = lambda a, d: (np.asarray(a, float)[orden] if a is not None else np.full(len(u), d))
    long_m, vel_kmh = tomar(long_m, 0.0), tomar(vel_kmh, 1.0)
    indptr = np.concatenate([[0], np.cumsum(np.bincount(u, minlength=n))])
    return Grafo(n=n, m=len(u), indptr=indptr, destino=v, lat=np.asarray(lat, float), lon=np.asarray(lon, float),
                 long_m=long_m, vel_kmh=vel_kmh, t0_s=3.6 * long_m / vel_kmh, peaje=tomar(peaje, 0.0),
                 energia=tomar(energia, 0.0), vmax_kmh=float(vmax_kmh if vmax_kmh else vel_kmh.max()),
                 origen=u, tipo=None if tipo is None else np.asarray(tipo)[orden],
                 carriles=None if carriles is None else np.asarray(carriles)[orden])


def heuristica_tiempo(g: Grafo, t: int, vmax_kmh: float) -> np.ndarray:
    """h(n) = d_hav(n,t)/vmax en segundos, para todos los nodos."""
    return haversine(g.lat, g.lon, g.lat[t], g.lon[t]) / (vmax_kmh / 3.6)


def heuristica_perezosa(g: Grafo, t: int, vmax_kmh: float):
    """h(n) escalar calculado bajo demanda (con cache); su costo entra al cronometro de A*."""
    cache, lt, gt, v = {}, float(g.lat[t]), float(g.lon[t]), vmax_kmh / 3.6
    lat, lon = g.lat, g.lon

    def h(n: int) -> float:
        r = cache.get(n)
        if r is None:
            la, lo = float(lat[n]), float(lon[n])
            a = math.sin((lt - la) / 2) ** 2 + math.cos(la) * math.cos(lt) * math.sin((gt - lo) / 2) ** 2
            r = cache[n] = 2 * R_TIERRA_M * math.asin(math.sqrt(min(1.0, a))) / v
        return r
    return h


def verificar_consistencia(g: Grafo, w: np.ndarray, vmax_kmh: float, destino: int | None = None, tol=1e-9) -> bool:
    """True si w_e >= d_hav(u,v)/vmax para toda arista (condicion fuerte, implica consistencia para
    cualquier destino). Si se da destino, tambien comprueba w_e >= h(u)-h(v)."""
    w = np.asarray(w, float)
    dh = haversine(g.lat[g.origen], g.lon[g.origen], g.lat[g.destino], g.lon[g.destino]) / (vmax_kmh / 3.6)
    ok = bool(np.all(w + tol >= dh))
    if destino is not None:
        h = heuristica_tiempo(g, destino, vmax_kmh)
        ok = ok and bool(np.all(w + tol >= h[g.origen] - h[g.destino]))
    return ok
