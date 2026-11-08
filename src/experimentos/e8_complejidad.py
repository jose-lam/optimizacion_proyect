"""E8 - Complejidad empirica: tiempo de Dijkstra (exploracion completa) vs (|V|+|E|) log|V|; ajuste log-log."""
import copy
import numpy as np, pandas as pd
from experimentos.comun import *
from busqueda import dijkstra, dijkstra_array
from datos.descargar_red import generar_red_sintetica, limpiar_grafo, a_grafo

TAM = [10, 15, 20, 30, 45, 60, 80, 100, 130]


def main():
    cfg = cargar_config(); rows = []
    for k in TAM:
        c = copy.deepcopy(cfg); c["areas"]["X"] = dict(filas=k, cols=k, paso_m=250)
        red, _ = limpiar_grafo(generar_red_sintetica("X", c)); g, _ = a_grafo(red, c)
        w = pesos_congestionados(g, cfg["semilla"])
        for _ in range(2):
            dijkstra(g, 0, -1, w)                                   # calentamiento
        _, t_heap = mediana_tiempo(lambda: dijkstra(g, 0, -1, w), 5)
        t_arr = np.nan
        if g.n <= 1000:
            _, t_arr = mediana_tiempo(lambda: dijkstra_array(g, 0, -1, w), 3)
        rows.append(dict(lado=k, V=g.n, E=g.m, VElogV=(g.n + g.m) * np.log(g.n), V2=g.n ** 2, t_heap=t_heap, t_array=t_arr))
    df = pd.DataFrame(rows)
    p = np.polyfit(np.log(df.VElogV), np.log(df.t_heap), 1)
    df["pendiente_loglog_heap"] = p[0]
    d2 = df.dropna(subset=["t_array"])
    df["pendiente_loglog_array_vs_V2"] = np.polyfit(np.log(d2.V2), np.log(d2.t_array), 1)[0] if len(d2) > 2 else np.nan
    guardar(df, "e8_complejidad"); print(df[["V", "E", "t_heap", "t_array"]], "\npendiente heap:", p[0])


if __name__ == "__main__":
    main()
