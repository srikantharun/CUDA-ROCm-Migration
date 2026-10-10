import numpy as np, sys
ok = True
# --- saxpy ---
n = 1 << 20; a = 2.0
x = np.ones(n, np.float32); y0 = np.arange(n, np.float32)
golden = (a * x.astype(np.float64) + y0.astype(np.float64))
got = np.fromfile("saxpy_out.bin", np.float32)
err = float(np.max(np.abs(got - golden)))
print(f"[saxpy]   max_abs_err={err:.3g}", "PASS" if np.allclose(got, golden, rtol=1e-4, atol=1e-2) else "FAIL")
ok &= np.allclose(got, golden, rtol=1e-4, atol=1e-2)
# --- softmax ---
rows, cols = 1024, 256
inp = np.fromfile("softmax_in.bin",  np.float32).reshape(rows, cols).astype(np.float64)
got = np.fromfile("softmax_out.bin", np.float32).reshape(rows, cols)
z = inp - inp.max(1, keepdims=True); e = np.exp(z); gold = e / e.sum(1, keepdims=True)
err = float(np.max(np.abs(got - gold)))
print(f"[softmax] max_abs_err={err:.3g}", "PASS" if err < 1e-5 else "FAIL")
ok &= err < 1e-5
sys.exit(0 if ok else 1)
