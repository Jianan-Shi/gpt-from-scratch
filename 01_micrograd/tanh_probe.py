import torch
from engine import Value

def probe(w_val):
    x = Value(2.0)
    w = Value(w_val)
    p = w * x
    h = p.tanh()
    L = h
    L.backward()
    return p.data, h.data, 1 - h.data**2, w.grad

print(f"{'w':<8}{'p':<10}{'h':<14}{'1-h^2':<14}{'w.grad':<14}")
for wv in [0.3, 1.0, 2.0, 3.0]:
    p, h, d, g = probe(wv)
    print(f"{wv:<8}{p:<10.3f}{h:<14.6f}{d:<14.8f}{g:<14.8f}")


import torch
from engine import Value

for wv in [0.3, 3.0]:
    # micrograd
    x, w = Value(2.0), Value(wv)
    h = (w * x).tanh()
    h.backward()
    mine = w.grad

    # PyTorch
    xt = torch.tensor(2.0)
    wt = torch.tensor(wv, requires_grad=True)
    ht = torch.tanh(wt * xt)
    ht.backward()
    ref = wt.grad.item()

    print(f"w={wv}: micrograd={mine:.10f}  torch={ref:.10f}  diff={abs(mine-ref):.2e}")