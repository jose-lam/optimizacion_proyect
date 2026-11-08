# Navegación y predicción de ETA (UTEC 2026) — implementación Parte 3

Ejecutar: `make pruebas` · `make datos` · `make experimentos` · `make figuras` (todo desde `PYTHONPATH=src`).
Los CSV crudos quedan en `resultados/crudo_csv/`; tablas y figuras se regeneran **solo** desde ellos (`experimentos/figuras.py`).

## Estado real de cada pieza
| Pieza | Estado |
|---|---|
| M2 `grafo.py`, M3 `busqueda.py` (Dijkstra, A*, array, bidireccional, dependiente del tiempo), M4 `costo.py`, `eta/huber.py`, `eta/baselines.py`, M6 `pipeline.py`, simulador de tráfico | Implementado y **ejecutado** |
| Golden tests (5 nodos, normalización, quiebre, gradiente de Huber -2.92 / 4.292) | **12/12 pasan**; total 27 pruebas ok |
| E0, E1, E2, E3, E4 (sin jerarquías), E5, E7, E8 | **Ejecutados** |
| E6: líneas base (persistencia, media histórica, ridge) | **Ejecutado** |
| E6: MLP L=0, GNN L=1,2,3, agregaciones, Huber vs cuadrática; búsqueda de hiperparámetros (`eta/gnn.py`, `eta/entrenar.py`) | Código escrito, **NO ejecutado** (no había `torch` ni red en el entorno de desarrollo). `tests/test_gnn.py` y la validación del gradiente con autograd se activan cuando exista torch |
| M1 con OSMnx (`desde_osmnx`, `descargar_red` con `fuente_red: osm`) | Código escrito, **NO probado** (sin red/osmnx) |
| Jerarquías de contracción, etiquetas de Pareto (Martins), reordenamiento por Ω(P) | **No implementados** (opcionales en el plan) |

## Supuestos y desviaciones (declarar en el informe)
- La red es **sintética tipo Lima** (malla con arteriales, autopistas de peaje y calles de un sentido), no OpenStreetMap: S=144, M=1 598, L=9 994 nodos. L no es Lima Metropolitana.
- Peajes (0.25 soles/arista de autopista), energía (α=0.12, β=1, γ=0.05; pendientes sintéticas) y 10 % de velocidades imputadas son valores asumidos.
- Tráfico **sintético**: ningún error de ETA es error sobre tráfico real.
- Estratos de distancia = terciles de la distancia en línea recta por área (en lugar de cortes fijos en km, que no caben en S/M).
- Muestras reducidas por costo de cómputo: E1 usa 1000 pares en S y M, 300 en L; E2–E4 usan 500 por estrato en S/M y 60 en L (E3/E4 máximo 150).
- E7 usa un estimador de velocidad actual con ruido lognormal (σ=0.15) e incidentes inyectados, no la GNN.
- Los tiempos se midieron en una máquina de 1 núcleo, sin ejecutar otras tareas a la vez.
