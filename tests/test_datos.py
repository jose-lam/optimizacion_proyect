import numpy as np
from experimentos.comun import cargar_config, grafo_area
from datos.trafico_sintetico import simular_trafico, lista_vecinos
from eta.datos import construir_dataset
from busqueda import verificar_fifo

cfg = cargar_config(); g, _ = grafo_area("S", cfg)
cfg2 = dict(cfg); cfg2["trafico"] = dict(cfg["trafico"], dias=10)
sim = simular_trafico(g, cfg2)


def test_division_temporal_sin_fuga():
    d = construir_dataset(g, sim, cfg2); i = d["idx"]
    dia = d["dia"]
    assert dia[i["train"]].max() < dia[i["val"]].min() <= dia[i["val"]].max() < dia[i["test"]].min()
    assert abs(d["X"][i["train"]].mean()) < 1e-3                   # estadisticas calculadas con entrenamiento
    assert ((d["taus"] + d["H"]) // d["spd"] == d["taus"] // d["spd"]).all()


def test_vecinos_simetricos_y_sin_autolazos():
    p = lista_vecinos(g); s = set(map(tuple, p.T.tolist()))
    assert all(a != b for a, b in s) and all((b, a) in s for a, b in s)


def test_simulador_fifo_y_positividad():
    assert (sim.v_true > 0).all() and sim.v_true.max() <= g.vmax_kmh + 1e-6
    taus = np.linspace(7.5 * 3600, 9 * 3600, 60)
    assert all(verificar_fifo(sim.tiempo_real, e, taus) for e in range(0, g.m, 40))
    assert abs(sim.tiempo_real(0, 3 * 3600) - g.t0_s[0] / (1 - 0)) < g.t0_s[0] * 3     # orden de magnitud razonable
