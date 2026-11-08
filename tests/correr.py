"""Ejecutor minimo: corre todas las funciones test_* (compatible con pytest si esta instalado)."""
import importlib.util, pathlib, sys, traceback
raiz = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(raiz.parent / "src")); sys.path.insert(0, str(raiz))
ok = fallos = omitidos = 0
for f in sorted(raiz.glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(f.stem, f); mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as e:
        print(f"OMITIDO {f.name}: {e}"); omitidos += 1; continue
    for nombre in sorted(dir(mod)):
        if nombre.startswith("test_"):
            try:
                getattr(mod, nombre)(); ok += 1; print("OK     ", f.stem, nombre)
            except ImportError as e:
                omitidos += 1; print("OMITIDO ", f.stem, nombre, f"({e})")
            except Exception:
                fallos += 1; print("FALLO  ", f.stem, nombre); traceback.print_exc()
print(f"\n{ok} ok, {fallos} fallos, {omitidos} omitidos (dependencia faltante)")
sys.exit(1 if fallos else 0)
