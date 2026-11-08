"""M3 - Dijkstra, A*, bidireccional, dependiente del tiempo (Parte 1, Alg. 1-3, sec. 3.6-3.7)."""
from __future__ import annotations
import heapq
import time
from dataclasses import dataclass
from typing import Callable, List
import numpy as np
from grafo import Grafo

INF = float("inf")


@dataclass
class Resultado:
    ruta: List[int]
    costo: float
    expandidos: int
    aristas_examinadas: int
    inserciones: int
    max_cola: int
    tiempo_s: float


def _pesos(w):
    return w if isinstance(w, list) else np.asarray(w, float).tolist()


def reconstruir(previo, t: int) -> List[int]:
    camino, n = [], t
    while n != -1:
        camino.append(n)
        n = previo[n]
    return camino[::-1]


def _res(previo, s, t, costo, exp, ar, ins, mq, t0):
    ruta = reconstruir(previo, t) if costo < INF and t >= 0 else []
    return Resultado(ruta, costo, exp, ar, ins, mq, time.perf_counter() - t0)


def dijkstra(g: Grafo, s: int, t: int, w) -> Resultado:
    """Heap binario con borrado perezoso. t<0 => explora todo el grafo."""
    wl, ip, dst = _pesos(w), g._ip, g._dst
    t0 = time.perf_counter()
    dist, prev = [INF] * g.n, [-1] * g.n
    dist[s] = 0.0
    q = [(0.0, s)]
    exp = ar = 0
    ins = mq = 1
    while q:
        d, u = heapq.heappop(q)
        if d > dist[u]:
            continue
        exp += 1
        if u == t:
            break
        for e in range(ip[u], ip[u + 1]):
            ar += 1
            v = dst[e]
            alt = d + wl[e]
            if alt < dist[v]:
                dist[v] = alt
                prev[v] = u
                heapq.heappush(q, (alt, v))
                ins += 1
                if len(q) > mq:
                    mq = len(q)
    return _res(prev, s, t, dist[t] if t >= 0 else 0.0, exp, ar, ins, mq, t0)


def astar(g: Grafo, s: int, t: int, w, h: Callable[[int], float]) -> Resultado:
    """Alg. 2: prioridad g+h, conjunto de cerrados. El tiempo incluye evaluar h."""
    wl, ip, dst = _pesos(w), g._ip, g._dst
    t0 = time.perf_counter()
    gs, prev, cerr = [INF] * g.n, [-1] * g.n, bytearray(g.n)
    gs[s] = 0.0
    q = [(h(s), s)]
    exp = ar = 0
    ins = mq = 1
    while q:
        _, u = heapq.heappop(q)
        if cerr[u]:
            continue
        cerr[u] = 1
        exp += 1
        if u == t:
            break
        gu = gs[u]
        for e in range(ip[u], ip[u + 1]):
            ar += 1
            v = dst[e]
            tent = gu + wl[e]
            if tent < gs[v]:
                gs[v] = tent
                prev[v] = u
                heapq.heappush(q, (tent + h(v), v))
                ins += 1
                if len(q) > mq:
                    mq = len(q)
    return _res(prev, s, t, gs[t], exp, ar, ins, mq, t0)


def dijkstra_array(g: Grafo, s: int, t: int, w) -> Resultado:
    """O(|V|^2): minimo por busqueda lineal. Solo para la prueba de complejidad."""
    wl, ip, dst = _pesos(w), g._ip, g._dst
    t0 = time.perf_counter()
    n = g.n
    dist, prev, cerr = [INF] * n, [-1] * n, [False] * n
    dist[s] = 0.0
    exp = ar = 0
    ins = 1
    while True:
        u, best = -1, INF
        for i in range(n):
            if not cerr[i] and dist[i] < best:
                u, best = i, dist[i]
        if u < 0:
            break
        cerr[u] = True
        exp += 1
        if u == t:
            break
        for e in range(ip[u], ip[u + 1]):
            ar += 1
            v = dst[e]
            if best + wl[e] < dist[v]:
                dist[v] = best + wl[e]
                prev[v] = u
                ins += 1
    return _res(prev, s, t, dist[t], exp, ar, ins, n, t0)


def bidireccional(g: Grafo, s: int, t: int, w) -> Resultado:
    """Dijkstra desde s (grafo) y desde t (transpuesto). Parada correcta: topF + topB >= mu."""
    wl, ip, dst = _pesos(w), g._ip, g._dst
    rip, rsrc, ridx = g.inversa()
    t0 = time.perf_counter()
    if s == t:
        return Resultado([s], 0.0, 1, 0, 1, 1, time.perf_counter() - t0)
    n = g.n
    dF, dB = [INF] * n, [INF] * n
    pF, pB = [-1] * n, [-1] * n
    dF[s] = 0.0
    dB[t] = 0.0
    qF, qB = [(0.0, s)], [(0.0, t)]
    mu, meet = INF, -1
    exp = ar = 0
    ins, mq = 2, 2

    def limpiar(q, d):
        while q and q[0][0] > d[q[0][1]]:
            heapq.heappop(q)

    while True:
        limpiar(qF, dF)
        limpiar(qB, dB)
        if not qF or not qB:
            break
        if qF[0][0] + qB[0][0] >= mu:
            break
        if len(qF) <= len(qB):
            d, u = heapq.heappop(qF)
            exp += 1
            for e in range(ip[u], ip[u + 1]):
                ar += 1
                v = dst[e]
                alt = d + wl[e]
                if alt < dF[v]:
                    dF[v] = alt
                    pF[v] = u
                    heapq.heappush(qF, (alt, v))
                    ins += 1
                    if alt + dB[v] < mu:
                        mu, meet = alt + dB[v], v
        else:
            d, u = heapq.heappop(qB)
            exp += 1
            for k in range(rip[u], rip[u + 1]):
                ar += 1
                v = rsrc[k]
                alt = d + wl[ridx[k]]
                if alt < dB[v]:
                    dB[v] = alt
                    pB[v] = u
                    heapq.heappush(qB, (alt, v))
                    ins += 1
                    if alt + dF[v] < mu:
                        mu, meet = alt + dF[v], v
        mq = max(mq, len(qF) + len(qB))
    ruta = []
    if mu < INF:
        ruta = reconstruir(pF, meet)
        n_ = pB[meet]
        while n_ != -1:
            ruta.append(n_)
            n_ = pB[n_]
    return Resultado(ruta, mu, exp, ar, ins, mq, time.perf_counter() - t0)


def verificar_fifo(tiempo_arista: Callable[[int, float], float], e: int, taus, tol=1e-9) -> bool:
    """tau1 + t_e(tau1) <= tau2 + t_e(tau2) para tau1 <= tau2 (ec. 8)."""
    sal = [tau + tiempo_arista(e, tau) for tau in sorted(taus)]
    return all(a <= b + tol for a, b in zip(sal, sal[1:]))


def dijkstra_td(g: Grafo, s: int, t: int, tau0: float, tiempo_arista: Callable[[int, float], float]) -> Resultado:
    """Dijkstra dependiente del tiempo (valido bajo FIFO). costo = tiempo de viaje (llegada - tau0)."""
    ip, dst = g._ip, g._dst
    t0 = time.perf_counter()
    arr, prev = [INF] * g.n, [-1] * g.n
    arr[s] = tau0
    q = [(tau0, s)]
    exp = ar = 0
    ins = mq = 1
    while q:
        d, u = heapq.heappop(q)
        if d > arr[u]:
            continue
        exp += 1
        if u == t:
            break
        for e in range(ip[u], ip[u + 1]):
            ar += 1
            v = dst[e]
            alt = d + tiempo_arista(e, d)
            if alt < arr[v]:
                arr[v] = alt
                prev[v] = u
                heapq.heappush(q, (alt, v))
                ins += 1
                mq = max(mq, len(q))
    return _res(prev, s, t, arr[t] - tau0 if arr[t] < INF else INF, exp, ar, ins, mq, t0)
