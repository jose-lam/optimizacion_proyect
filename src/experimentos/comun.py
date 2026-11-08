"""Utilidades comunes de los experimentos."""
import pathlib, sys, time, platform, json
import numpy as np, pandas as pd, yaml

RAIZ = pathlib.Path(__file__).resolve().parents[2]
CSV = RAIZ / "resultados" / "crudo_csv"
TAB = RAIZ / "resultados" / "tablas"
FIG = RAIZ / "resultados" / "figuras"


def cargar_config():
    with open(RAIZ / "config.yaml") as f:
        return yaml.safe_load(f)


_cache = {}
def grafo_area(area, cfg):
    from datos.descargar_red import descargar_red
    if area not in _cache:
        _cache[area] = descargar_red(area, cfg)
    return _cache[area]


def pares_estratificados(g, n_por_estrato, rng, n_estratos=3):
    """Pares uniformes (s != t) estratificados por terciles de la distancia en linea recta."""
    from grafo import haversine
    k = n_por_estrato * n_estratos * 12
    s = rng.integers(0, g.n, k); t = rng.integers(0, g.n, k)
    ok = s != t; s, t = s[ok], t[ok]
    d = haversine(g.lat[s], g.lon[s], g.lat[t], g.lon[t]) / 1000.0
    cortes = np.quantile(d, np.linspace(0, 1, n_estratos + 1)); cortes[-1] += 1
    nombres = ["corta", "media", "larga"]
    out = []
    for i in range(n_estratos):
        idx = np.where((d >= cortes[i]) & (d < cortes[i + 1]))[0][:n_por_estrato]
        out += [(int(s[j]), int(t[j]), nombres[i], float(d[j])) for j in idx]
    return out


def mediana_tiempo(fn, reps=3, calentar=0):
    """Ejecuta fn() -> Resultado varias veces; devuelve (primer resultado, mediana de tiempo_s)."""
    import gc
    for _ in range(calentar):
        fn()
    gc.disable()
    try:
        rs = [fn() for _ in range(reps)]
    finally:
        gc.enable()
    return rs[0], float(np.median([r.tiempo_s for r in rs]))


def guardar(df, nombre):
    CSV.mkdir(parents=True, exist_ok=True)
    df.to_csv(CSV / f"{nombre}.csv", index=False)
    print(f"[{nombre}] {len(df)} filas -> {CSV / (nombre + '.csv')}")


def meta():
    import numpy, scipy, pandas, networkx
    d = dict(python=platform.python_version(), numpy=numpy.__version__, scipy=scipy.__version__, pandas=pandas.__version__,
             networkx=networkx.__version__, maquina=platform.platform(), procesador=platform.processor())
    CSV.mkdir(parents=True, exist_ok=True)
    (CSV / "entorno.json").write_text(json.dumps(d, indent=1))


def bootstrap_ic(x, estad=np.mean, B=2000, seed=42):
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    bs = [estad(r.choice(x, len(x))) for _ in range(B)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def pesos_congestionados(g, semilla=42):
    """Pesos de prueba (segundos): t0 * U(1, 1.8) por arista (congestion fija). Siempre >= t0 >= d/vmax."""
    r = np.random.default_rng(semilla)
    return (g.t0_s * r.uniform(1.0, 1.8, g.m)).tolist()


def a_networkx(g, w):
    import networkx as nx
    G = nx.DiGraph()
    G.add_weighted_edges_from(zip(g.origen.tolist(), g.destino.tolist(), w))
    return G
