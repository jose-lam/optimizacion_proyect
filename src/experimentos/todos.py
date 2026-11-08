"""Ejecuta E0-E8 en orden."""
from experimentos import e0_eda, e1_correctitud, e2_dijkstra_vs_astar, e3_heuristica, e4_bidireccional, e5_multicriterio, e6_eta, e7_recalculo, e8_complejidad
for m in (e0_eda, e1_correctitud, e2_dijkstra_vs_astar, e3_heuristica, e4_bidireccional, e5_multicriterio, e6_eta, e7_recalculo, e8_complejidad):
    print("=" * 20, m.__name__); m.main()
