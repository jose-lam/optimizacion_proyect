PY=PYTHONPATH=src python3
datos:         ; $(PY) -m experimentos.e0_eda
pruebas:       ; $(PY) tests/correr.py
experimentos:  ; $(PY) -m experimentos.todos
figuras:       ; $(PY) -m experimentos.figuras
