"""M1 - Obtencion y limpieza de la red vial.
 - descargar_red(area, cfg): OSMnx si cfg['fuente_red']=='osm' (requiere red; NO probado en este entorno),
   o red sintetica tipo Lima (malla con arteriales, autopistas de peaje, calles de un sentido).
 - limpiar_grafo(): mayor componente fuertemente conexa, sin bucles, sin aristas paralelas.
"""
from __future__ import annotations
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from grafo import construir_grafo, haversine, Grafo
from datos.atributos import (VEL_POR_TIPO, imputar_atributos, peaje_por_arista, energia_por_arista)


def _tipo_linea(i: int) -> int:
    if i % 12 == 6:
        return 3
    if i % 4 == 0:
        return 2
    return 1 if i % 2 == 0 else 0


def generar_red_sintetica(area: str, cfg: dict, semilla: int | None = None) -> dict:
    rng = np.random.default_rng(cfg["semilla"] if semilla is None else semilla)
    a = cfg["areas"][area]
    nf, nc, paso = a["filas"], a["cols"], a["paso_m"]
    lat0, lon0 = np.radians(cfg["centro"]["lat"]), np.radians(cfg["centro"]["lon"])
    ii, jj = np.meshgrid(np.arange(nf), np.arange(nc), indexing="ij")
    dy = (ii + rng.uniform(-0.15, 0.15, ii.shape)) * paso
    dx = (jj + rng.uniform(-0.15, 0.15, jj.shape)) * paso
    lat = (lat0 + dy / 6_371_000.0).ravel()
    lon = (lon0 + dx / (6_371_000.0 * np.cos(lat0))).ravel()
    idn = lambda i, j: i * nc + j
    u, v, tipo = [], [], []
    for i in range(nf):
        for j in range(nc):
            if j + 1 < nc:
                u.append(idn(i, j)); v.append(idn(i, j + 1)); tipo.append(_tipo_linea(i))
            if i + 1 < nf:
                u.append(idn(i, j)); v.append(idn(i + 1, j)); tipo.append(_tipo_linea(j))
    u, v, tipo = np.array(u), np.array(v), np.array(tipo)
    keep = rng.random(len(u)) > 0.03                      # 3 % de calles ausentes
    u, v, tipo = u[keep], v[keep], tipo[keep]
    uni = (tipo == 0) & (rng.random(len(u)) < 0.30)       # calles residenciales de un sentido
    inv = uni & (rng.random(len(u)) < 0.5)
    u2, v2 = np.where(inv, v, u), np.where(inv, u, v)
    U = np.concatenate([u2, v[~uni]]); V = np.concatenate([v2, u[~uni]])
    T = np.concatenate([tipo, tipo[~uni]])
    d_hav = haversine(lat[U], lon[U], lat[V], lon[V])
    long_m = d_hav * rng.uniform(1.0, 1.15, len(U))       # longitud por carretera >= linea recta
    vel = np.array([VEL_POR_TIPO[int(t)] for t in T], float)
    vel[rng.random(len(U)) < 0.10] = np.nan               # 10 % de velocidades faltantes (para imputar)
    carriles = np.array([1, 1, 2, 3])[T]
    pend = np.where(rng.random(len(U)) < 0.3, rng.uniform(0.0, 0.05, len(U)), 0.0)  # pendiente SINTETICA
    return dict(n=nf * nc, u=U, v=V, lat=lat, lon=lon, long_m=long_m, vel=vel, tipo=T, carriles=carriles, pend=pend)


def limpiar_grafo(red: dict) -> tuple[dict, dict]:
    """Mayor SCC, sin bucles, aristas paralelas u->v: conserva la de menor tiempo."""
    n, u, v = red["n"], red["u"], red["v"]
    stats = {"n_inicial": n, "m_inicial": len(u)}
    ok = u != v
    stats["bucles"] = int((~ok).sum())
    cols = {k: red[k][ok] for k in ("u", "v", "long_m", "vel", "tipo", "carriles", "pend")}
    t = cols["long_m"] / np.where(np.isfinite(cols["vel"]), cols["vel"], 30.0)
    orden = np.lexsort((t, cols["v"], cols["u"]))
    cols = {k: x[orden] for k, x in cols.items()}
    primero = np.ones(len(cols["u"]), bool)
    primero[1:] = (cols["u"][1:] != cols["u"][:-1]) | (cols["v"][1:] != cols["v"][:-1])
    stats["paralelas_descartadas"] = int((~primero).sum())
    cols = {k: x[primero] for k, x in cols.items()}
    A = coo_matrix((np.ones(len(cols["u"])), (cols["u"], cols["v"])), shape=(n, n))
    nc, lab = connected_components(A, directed=True, connection="strong")
    mayor = np.argmax(np.bincount(lab))
    nodos = np.where(lab == mayor)[0]
    nuevo = -np.ones(n, int); nuevo[nodos] = np.arange(len(nodos))
    m = (nuevo[cols["u"]] >= 0) & (nuevo[cols["v"]] >= 0)
    out = {k: x[m] for k, x in cols.items()}
    out["u"], out["v"] = nuevo[out["u"]], nuevo[out["v"]]
    out.update(n=len(nodos), lat=red["lat"][nodos], lon=red["lon"][nodos])
    stats.update(n_final=len(nodos), m_final=len(out["u"]), nodos_descartados=n - len(nodos))
    return out, stats


def a_grafo(red: dict, cfg: dict) -> tuple[Grafo, dict]:
    vel, pct = imputar_atributos(red["vel"], red["tipo"])
    g = construir_grafo(red["n"], red["u"], red["v"], red["lat"], red["lon"], red["long_m"], vel,
                        peaje=peaje_por_arista(red["tipo"], cfg),
                        energia=energia_por_arista(red["long_m"], vel, red["pend"], cfg),
                        tipo=red["tipo"], carriles=red["carriles"])
    return g, {"pct_vel_imputada": pct}


def desde_osmnx(G):
    """Convierte un MultiDiGraph de OSMnx al dict 'red' (NO probado aqui: sin red ni osmnx)."""
    import osmnx as ox
    nod, ari = ox.graph_to_gdfs(G)
    idx = {nid: i for i, nid in enumerate(nod.index)}
    u = np.array([idx[a] for a, b, k in ari.index]); v = np.array([idx[b] for a, b, k in ari.index])
    def vel(x):
        try:
            x = x[0] if isinstance(x, list) else x
            return float(str(x).split()[0])
        except Exception:
            return np.nan
    mapa = {"motorway": 3, "trunk": 3, "primary": 2, "secondary": 1}
    def tp(x):
        x = x[0] if isinstance(x, list) else x
        return mapa.get(x, 0)
    tipo = np.array([tp(x) for x in ari["highway"]])
    return dict(n=len(nod), u=u, v=v, lat=np.radians(nod["y"].values), lon=np.radians(nod["x"].values),
                long_m=ari["length"].values.astype(float), vel=np.array([vel(x) for x in ari.get("maxspeed", [None] * len(ari))]),
                tipo=tipo, carriles=np.ones(len(u), int), pend=np.zeros(len(u)))


def descargar_red(area: str, cfg: dict):
    """Devuelve (Grafo, estadisticas_de_limpieza)."""
    if cfg.get("fuente_red") == "osm":
        import osmnx as ox
        G = ox.graph_from_place(cfg["areas"][area]["lugar"], network_type="drive")
        red = desde_osmnx(G)
    else:
        red = generar_red_sintetica(area, cfg)
    red, st = limpiar_grafo(red)
    g, st2 = a_grafo(red, cfg)
    st.update(st2)
    st["area"] = area
    return g, st


def validar_grafo(g: Grafo) -> dict:
    """Validaciones automaticas de M1."""
    from scipy.sparse import csr_matrix
    nc, _ = connected_components(csr_matrix((np.ones(g.m), g.destino, g.indptr), shape=(g.n, g.n)), directed=True, connection="strong")
    return {"fuertemente_conexo": nc == 1, "costos_no_negativos": bool(np.all(g.t0_s >= 0) and np.all(np.isfinite(g.t0_s))),
            "vel_le_vmax": bool(np.all(g.vel_kmh <= g.vmax_kmh + 1e-9)),
            "long_ge_recta": bool(np.all(g.long_m + 1e-6 >= __import__("grafo").haversine(g.lat[g.origen], g.lon[g.origen], g.lat[g.destino], g.lon[g.destino])))}
