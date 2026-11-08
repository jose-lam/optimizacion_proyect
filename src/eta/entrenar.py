"""M5 - Bucle de entrenamiento (Adam, weight_decay = 2*lambda), parada temprana y busqueda aleatoria de hiperparametros.
Requiere torch (no disponible donde se genero el repositorio: codigo sin ejecutar)."""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from eta.gnn import GNNETA

_cache_pares = {}


def _pares_lote(pares, b, m):
    k = (b, m, pares.shape[1])
    if k not in _cache_pares:
        _cache_pares[k] = torch.cat([pares + i * m for i in range(b)], 1)
    return _cache_pares[k]


def _perdida(pred, y, delta, tipo):
    return F.huber_loss(pred, y, delta=delta) if tipo == "huber" else 0.5 * F.mse_loss(pred, y)


def predecir(modelo, X, pares, lote=64):
    modelo.eval(); out = []
    m = X.shape[1]
    with torch.no_grad():
        for i in range(0, len(X), lote):
            xb = torch.as_tensor(X[i:i + lote]); b = xb.shape[0]
            out.append(modelo(xb.reshape(b * m, -1), _pares_lote(pares, b, m)).reshape(b, m).numpy())
    return np.concatenate(out)


def entrenar(modelo, datos, pares, eta, lam, delta, epocas, paciencia, semilla, lote=16, perdida="huber"):
    """Con L(theta)=mean(rho)+lam||theta||^2 el gradiente de la penalizacion es 2*lam*theta = weight_decay*theta => wd=2*lam."""
    torch.manual_seed(semilla); rng = np.random.default_rng(semilla)
    opt = torch.optim.Adam(modelo.parameters(), lr=eta, weight_decay=2 * lam)
    X, y, idx = datos["X"], datos["y"], datos["idx"]
    m = X.shape[1]
    mejor, mejor_estado, sin_mejora, hist = np.inf, None, 0, []
    for ep in range(epocas):
        modelo.train()
        orden = rng.permutation(idx["train"]); tr = []
        for i in range(0, len(orden), lote):
            ib = orden[i:i + lote]; b = len(ib)
            xb = torch.as_tensor(X[ib]).reshape(b * m, -1); yb = torch.as_tensor(y[ib]).reshape(-1)
            opt.zero_grad()
            l = _perdida(modelo(xb, _pares_lote(pares, b, m)), yb, delta, perdida)
            l.backward(); opt.step(); tr.append(float(l))
        pv = predecir(modelo, X[idx["val"]], pares)
        lv = float(_perdida(torch.as_tensor(pv), torch.as_tensor(y[idx["val"]]), delta, perdida))
        hist.append(dict(epoca=ep, perdida_train=float(np.mean(tr)), perdida_val=lv))
        if lv < mejor - 1e-6:
            mejor, mejor_estado, sin_mejora = lv, copy.deepcopy(modelo.state_dict()), 0
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    modelo.load_state_dict(mejor_estado)
    return hist


def busqueda_hiperparametros(datos, pares, d_in, n_trials, semilla, epocas_cortas=12, delta=1.0):
    """Busqueda aleatoria con presupuesto fijo; rangos del plan: eta in [1e-4,1e-2], lam in {0,1e-5,1e-4}, L in {1,2,3}, d in {16,32,64}."""
    rng = np.random.default_rng(semilla); res = []
    for k in range(n_trials):
        c = dict(eta=float(10 ** rng.uniform(-4, -2)), lam=float(rng.choice([0, 1e-5, 1e-4])), L=int(rng.choice([1, 2, 3])), d=int(rng.choice([16, 32, 64])))
        torch.manual_seed(semilla + k)
        mod = GNNETA(d_in, c["d"], c["L"], "mean")
        h = entrenar(mod, datos, pares, c["eta"], c["lam"], delta, epocas_cortas, 4, semilla + k)
        res.append(dict(**c, delta=delta, mejor_val=min(x["perdida_val"] for x in h)))
    return sorted(res, key=lambda r: r["mejor_val"])
