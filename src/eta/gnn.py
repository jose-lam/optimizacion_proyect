"""M5 - Paso de mensajes en el grafo de lineas (Parte 1, ec. 12-13) con PyTorch puro (index_add_ / scatter_reduce).
NOTA: este modulo requiere torch; en el entorno donde se genero el repositorio torch no estaba disponible, por lo que
solo se verifico sintacticamente. Las pruebas tests/test_gnn.py lo ejecutan donde torch exista."""
import torch
import torch.nn as nn


def agregar(msgs, dest, n, modo="mean"):
    """Agregacion invariante a permutaciones de msgs (P,d) hacia n aristas segun dest (P,)."""
    d = msgs.shape[1]
    if modo in ("sum", "mean"):
        out = torch.zeros(n, d, dtype=msgs.dtype, device=msgs.device).index_add_(0, dest, msgs)
        if modo == "mean":
            cnt = torch.zeros(n, dtype=msgs.dtype, device=msgs.device).index_add_(0, dest, torch.ones(len(dest), dtype=msgs.dtype, device=msgs.device))
            out = out / cnt.clamp(min=1.0)[:, None]
        return out
    if modo == "max":
        base = torch.zeros(n, d, dtype=msgs.dtype, device=msgs.device)
        return base.scatter_reduce(0, dest[:, None].expand(-1, d), msgs, reduce="amax", include_self=False)
    raise ValueError(modo)


class CapaMensaje(nn.Module):
    """m_e = AGG_{e' in N(e)} phi([h_e, h_e']);  h_e <- psi([h_e, m_e])."""
    def __init__(self, d: int, agregacion: str = "mean"):
        super().__init__()
        self.agregacion = agregacion
        self.phi = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU())
        self.psi = nn.Sequential(nn.Linear(2 * d, d), nn.ReLU())

    def forward(self, h, pares):             # pares: (2,P) = (indices de e, indices de e')
        dest, src = pares[0], pares[1]
        msg = self.phi(torch.cat([h[dest], h[src]], 1))
        m = agregar(msg, dest, h.shape[0], self.agregacion)
        return self.psi(torch.cat([h, m], 1))


class GNNETA(nn.Module):
    """Codificador -> L capas de paso de mensajes -> capa de salida. L=0 es el MLP sin grafo."""
    def __init__(self, d_in: int, d: int, L: int, agregacion: str = "mean"):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(d_in, d), nn.ReLU())
        self.capas = nn.ModuleList([CapaMensaje(d, agregacion) for _ in range(L)])
        self.salida = nn.Linear(d, 1)

    def forward(self, x, pares):
        h = self.enc(x)
        for c in self.capas:
            h = c(h, pares)
        return self.salida(h).squeeze(-1)
