"""Genera figuras y tablas (markdown) SOLO a partir de resultados/crudo_csv."""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from experimentos.comun import CSV, FIG, TAB, bootstrap_ic


def md(df, nombre, fmt="%.4g"):
    TAB.mkdir(parents=True, exist_ok=True)
    try:
        txt = df.to_markdown(index=False, floatfmt=".4g")
    except ImportError:                      # tabulate no instalado
        txt = df.to_string(index=False)
    (TAB / f"{nombre}.md").write_text(txt)
    df.to_csv(TAB / f"{nombre}.csv", index=False)


def leer(n):
    p = CSV / f"{n}.csv"
    return pd.read_csv(p) if p.exists() else None


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    e0 = leer("e0_tamanos")
    if e0 is not None:
        md(e0[["area", "nodos", "aristas", "grado_medio", "long_total_km", "vmax_kmh", "pct_vel_imputada", "pct_peaje"]], "t12_tamanos")
        a, gr = leer("e0_aristas"), leer("e0_grados")
        fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
        for ar in "SML":
            ax[0].hist(a[a.area == ar].long_m, bins=40, histtype="step", label=ar, density=True)
            ax[2].hist(gr[gr.area == ar].grado_salida, bins=range(0, 9), histtype="step", label=ar, density=True)
        ax[1].hist([a[a.tipo == t].vel_kmh for t in range(4)], bins=[25, 35, 50, 70, 90], label=["resid", "secund", "arterial", "autopista"])
        for x, t in zip(ax, ["Longitud de arista (m)", "Velocidad (km/h)", "Grado de salida"]): x.set_xlabel(t); x.legend()
        fig.tight_layout(); fig.savefig(FIG / "e0_histogramas.png", dpi=130); plt.close(fig)
    e1 = leer("e1_correctitud")
    if e1 is not None:
        md(e1.groupby(["area", "algoritmo"]).agg(pares=("dif_rel", "size"), dif_rel_max=("dif_rel", "max"), pct_coinciden=("dif_rel", lambda x: 100 * (x < 1e-9).mean())).reset_index(), "t13_e1")
    e2 = leer("e2_dijkstra_vs_astar")
    if e2 is not None:
        filas = []
        for (ar, es), d in e2.groupby(["area", "estrato"]):
            r_exp = d.exp_ast / d.exp_dij; r_t = d.t_ast / d.t_dij
            w_e = stats.wilcoxon(d.exp_dij, d.exp_ast).pvalue if (d.exp_dij != d.exp_ast).any() else np.nan
            w_t = stats.wilcoxon(d.t_dij, d.t_ast).pvalue
            filas.append(dict(area=ar, estrato=es, n=len(d), exp_dij=d.exp_dij.mean(), exp_ast=d.exp_ast.mean(), razon_exp=r_exp.mean(),
                              razon_exp_IC95=str(tuple(round(v, 3) for v in bootstrap_ic(r_exp))), t_dij_ms=1e3 * d.t_dij.mean(), t_ast_ms=1e3 * d.t_ast.mean(),
                              razon_t=r_t.mean(), razon_t_IC95=str(tuple(round(v, 3) for v in bootstrap_ic(r_t))), wilcoxon_p_exp=w_e, wilcoxon_p_t=w_t,
                              mismo_costo_pct=100 * d.mismo_costo.mean()))
        md(pd.DataFrame(filas), "t13_e2")
        ords = {"corta": 0, "media": 1, "larga": 2}
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.4))
        for k, (col_d, col_a, tit) in enumerate([("exp_dij", "exp_ast", "Nodos expandidos"), ("t_dij", "t_ast", "Tiempo (s)")]):
            for i, ar in enumerate("SML"):
                dd = e2[e2.area == ar]
                for j, es in enumerate(["corta", "media", "larga"]):
                    x = dd[dd.estrato == es]
                    ax[k].bar(i * 3.2 + j * 1.0 - .2, x[col_d].mean(), .35, color="C0"); ax[k].bar(i * 3.2 + j * 1.0 + .2, x[col_a].mean(), .35, color="C1")
            ax[k].set_xticks([i * 3.2 + 1 for i in range(3)]); ax[k].set_xticklabels(["S", "M", "L"]); ax[k].set_title(tit + " (azul=Dijkstra, naranja=A*)"); ax[k].set_yscale("log")
        fig.tight_layout(); fig.savefig(FIG / "e2_dijkstra_vs_astar.png", dpi=130); plt.close(fig)
    e3 = leer("e3_heuristica")
    if e3 is not None:
        t = e3.groupby(["area", "factor"]).agg(expandidos=("expandidos", "mean"), pct_no_optimas=("optima", lambda x: 100 * (1 - x.mean())), exceso_costo_pct=("exceso_rel", lambda x: 100 * x.mean()), exceso_max_pct=("exceso_rel", lambda x: 100 * x.max())).reset_index()
        md(t, "t13_e3")
        fig, ax = plt.subplots(1, 2, figsize=(9, 3.3))
        for ar in "SML":
            d = t[(t.area == ar) & (t.factor > 0)]; ax[0].plot(d.factor, d.expandidos, "o-", label=ar); ax[1].plot(d.factor, d.pct_no_optimas, "o-", label=ar)
        ax[0].set_xlabel("factor sobre vmax real (h = d/(factor*vmax))  [factor<1: sobreestima]"); ax[0].set_ylabel("nodos expandidos"); ax[1].set_ylabel("% rutas no optimas"); ax[1].set_xlabel("factor"); ax[0].legend()
        fig.tight_layout(); fig.savefig(FIG / "e3_heuristica.png", dpi=130); plt.close(fig)
    e4 = leer("e4_bidireccional")
    if e4 is not None:
        md(e4.groupby(["area", "algoritmo"]).agg(expandidos=("expandidos", "mean"), aristas=("aristas", "mean"), tiempo_ms=("tiempo_s", lambda x: 1e3 * x.mean())).reset_index(), "t13_e4")
    e5 = leer("e5_quiebres")
    if e5 is not None:
        ej, pm = leer("e5_ejemplo_parte1"), leer("e5_percentiles_vs_minmax")
        md(ej, "t14_e5_ejemplo")
        resumen = pd.DataFrame([dict(n_quiebres=len(e5), dif_max=e5.dif.max() if len(e5) else np.nan, pct_pares_misma_ruta_pct_vs_minmax=100 * pm.misma_ruta.mean(),
                                     z_t_medio_percentil=pm.z_t_percentil.mean(), z_t_medio_minmax=pm.z_t_minmax.mean(), rutas_distintas_media=pm.n_rutas_distintas.mean())])
        md(resumen, "t14_e5_resumen")
        if len(e5):
            fig, ax = plt.subplots(figsize=(4, 4)); ax.scatter(e5.w_analitico, e5.w_numerico, s=10); lim = [0, e5[["w_analitico", "w_numerico"]].max().max()]
            ax.plot(lim, lim, "k--"); ax.set_xlabel("w_p* analitico"); ax.set_ylabel("w_p numerico (biseccion)"); fig.tight_layout(); fig.savefig(FIG / "e5_quiebre.png", dpi=130); plt.close(fig)
    e6 = leer("e6_resultados")
    if e6 is not None:
        md(e6, "t14_e6")
    c = leer("e6_curvas")
    if c is not None:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        for n, d in c.groupby("modelo"): ax.plot(d.epoca, d.perdida_val, label=n)
        ax.set_xlabel("epoca"); ax.set_ylabel("perdida validacion"); ax.legend(fontsize=6); fig.tight_layout(); fig.savefig(FIG / "e6_curvas.png", dpi=130); plt.close(fig)
    e7 = leer("e7_recalculo")
    if e7 is not None:
        t = e7.groupby("rho_pct").agg(tiempo_total_s=("tiempo_total", "mean"), cambios=("cambios", "mean"), ahorro_vs_sin_s=("ahorro_vs_sin", "mean"), brecha_oraculo_s=("brecha_oraculo", "mean"), t_recalculo_ms=("t_recalculo_s", lambda x: 1e3 * x.mean())).reset_index()
        md(t, "t14_e7")
        fig, ax = plt.subplots(figsize=(4.5, 3.5)); ax.plot(t.cambios, t.ahorro_vs_sin_s, "o-")
        for _, r in t.iterrows(): ax.annotate(f"rho={r.rho_pct:g}%", (r.cambios, r.ahorro_vs_sin_s), fontsize=7)
        ax.set_xlabel("cambios de ruta por viaje"); ax.set_ylabel("ahorro vs sin recalculo (s)"); fig.tight_layout(); fig.savefig(FIG / "e7_compromiso.png", dpi=130); plt.close(fig)
    e8 = leer("e8_complejidad")
    if e8 is not None:
        fig, ax = plt.subplots(figsize=(4.5, 3.6)); ax.loglog(e8.VElogV, e8.t_heap, "o-", label=f"heap (pendiente {e8.pendiente_loglog_heap.iloc[0]:.2f})")
        d = e8.dropna(subset=["t_array"]); ax.loglog(d.V2, d.t_array, "s--", label="arreglo vs |V|^2")
        ax.set_xlabel("(|V|+|E|) log|V|  (o |V|^2)"); ax.set_ylabel("tiempo (s)"); ax.legend(); fig.tight_layout(); fig.savefig(FIG / "e8_loglog.png", dpi=130); plt.close(fig)
        md(e8[["lado", "V", "E", "t_heap", "t_array"]], "t13_e8")
    print("figuras/tablas generadas en", FIG, TAB)


if __name__ == "__main__":
    main()
