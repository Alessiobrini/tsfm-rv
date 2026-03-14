"""Quick smoke test for Moirai-MoE with short context lengths."""
import sys
import numpy as np

sys.path.insert(0, "code")
from models.foundation import MoiraiMoEModel

ctx_raw = np.random.rand(600).astype(np.float32) * 1e-4

for ctx_len in [128, 256, 512]:
    m = MoiraiMoEModel(context_length=ctx_len, device="cpu")
    for h in [1, 5, 22]:
        try:
            out = m.predict(ctx_raw, horizon=h)
            print(f"OK  ctx={ctx_len} h={h}: point={out.point[:3]}")
        except Exception as e:
            print(f"FAIL ctx={ctx_len} h={h}: {e}")
