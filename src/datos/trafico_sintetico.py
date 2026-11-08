"""M5 - Simulador sintetico de velocidades v_e(tau) = v0_e (1 - kappa_e(tau)).
DATOS SINTETICOS: ningun resultado sobre ellos es un error sobre trafico real."""
from __future__ import annotations
import numpy as np
from scipy.sparse import csr_matrix


def lista_vecinos(g):
    """Grafo de lineas: pares (dest, src) de aristas que comparten un nodo (e != e'). Devuelve array (2, P)."""
    rip, rsrc, ridx = g.inversa()
    salientes = [list(range(g._ip[u], g._ip[u + 1])) for u in range(g.n)]
    pares = set()
    for u in range(g.n):
        inc = salientes[u] + ridx[rip[u]:rip[u + 1]]
        for a in inc:
            for b in inc:
                if a != b:
                    pares.add(a * g.m + b)
    arr = np.array(sorted(pares), dtype=np.int64)
    return np.stack([arr // g.m, arr % g.m]).astype(np.int64)


class Simulador:
    def __init__(self, g, v_true, v_obs, paso_s):
        self.g, self.v_true, self.v_obs, self.paso_s = g, v_true, v_obs, paso_s
        self.T = v_true.shape[0]
        self.overlay = {}          # e -> [(paso0, paso1, factor)] incidentes reactivos inyectados

    @property
    def t_inst(self):
        return 3.6 * self.g.long_m[None, :] / self.v_true

    def añadir_incidente(self, e, t0_s, t1_s, factor):
        self.overlay.setdefault(e, []).append((int(t0_s // self.paso_s), int(t1_s // self.paso_s), factor))

    def limpiar_incidentes(self):
        self.overlay = {}

    def vel(self, e, paso):
        paso = min(max(paso, 0), self.T - 1)
        v = float(self.v_true[paso, e])
        for p0, p1, f in self.overlay.get(e, ()):
            if p0 <= paso <= p1:
                v *= f
        return v

    def vel_vector(self, tau_s):
        paso = min(max(int(tau_s // self.paso_s), 0), self.T - 1)
        v = self.v_true[paso].astype(float).copy()
        for e, lst in self.overlay.items():
            for p0, p1, f in lst:
                if p0 <= paso <= p1:
                    v[e] *= f
        return v

    def tiempo_real(self, e, tau):
        """Tiempo (s) de recorrer la arista e entrando en tau, integrando la velocidad (cumple FIFO)."""
        resto, t = float(self.g.long_m[e]), float(tau)
        while True:
            paso = int(t // self.paso_s)
            v = self.vel(e, paso) / 3.6
            quedan = (paso + 1) * self.paso_s - t
            if paso >= self.T - 1 or v * quedan >= resto:
                return t + resto / v - tau
            resto -= v * quedan
            t = (paso + 1) * self.paso_s


def simular_trafico(g, cfg, semilla=None) -> Simulador:
    rng = np.random.default_rng(cfg["semilla"] if semilla is None else semilla)
    tc = cfg["trafico"]
    spd = 1440 // tc["paso_min"]
    T = tc["dias"] * spd
    pares = lista_vecinos(g)
    A = csr_matrix((np.ones(pares.shape[1]), (pares[0], pares[1])), shape=(g.m, g.m))
    grado = np.asarray(A.sum(1)).ravel().clip(min=1)
    An = csr_matrix(A.multiply(1 / grado[:, None]))
    hora = (np.arange(T) % spd) * tc["paso_min"] / 60.0
    perfil = 0.05 + 0.45 * np.exp(-0.5 * ((hora - 8) / 1.2) ** 2) + 0.50 * np.exp(-0.5 * ((hora - 18) / 1.5) ** 2)
    sens = np.array([0.5, 0.7, 1.0, 0.8])[g.tipo]
    kappa = perfil[:, None] * sens[None, :]
    eps, prev = np.empty((T, g.m)), np.zeros(g.m)
    for t in range(T):                                   # ruido con correlacion espacial y temporal
        prev = 0.9 * prev + 0.3 * (An @ rng.standard_normal(g.m)) + 0.1 * rng.standard_normal(g.m)
        eps[t] = prev
    kappa = kappa + 0.06 * eps
    A2 = An @ An
    for d in range(tc["dias"]):                          # incidentes raros: caidas bruscas (valores atipicos)
        for _ in range(tc["incidentes_por_dia"]):
            e0, ini, dur = int(rng.integers(g.m)), d * spd + int(rng.integers(spd - 20)), int(rng.integers(6, 19))
            kappa[ini:ini + dur, e0] += 0.6
            vec1, vec2 = An[e0].indices, A2[e0].indices
            kappa[ini:ini + dur][:, vec1] += 0.3
            kappa[ini:ini + dur][:, vec2] += 0.1
    kappa = np.clip(kappa, 0.0, 0.9)
    v0 = g.vel_kmh[None, :]
    v_true = (v0 * (1 - kappa)).astype(np.float32)
    v_obs = np.clip(v_true * (1 + 0.03 * rng.standard_normal(v_true.shape)), 1.0, None).astype(np.float32)
    return Simulador(g, v_true, v_obs, tc["paso_min"] * 60.0)
