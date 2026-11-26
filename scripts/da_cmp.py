"""Compare board depth outputs with the float ONNX reference saved by da_ref.py.
   python3 scripts/da_cmp.py board_out/pcq board_out/i16"""
import numpy as np, sys, glob
from scipy.stats import spearmanr
n = len(glob.glob('da/test/d_*.ref.npy'))
for d in sys.argv[1:]:
    c = []; sp = []
    for k in range(n):
        r = np.load(f'da/test/d_{k:02d}.ref.npy'); l = np.fromfile(f'{d}/d_{k:02d}.bin', np.float32)
        c.append(np.corrcoef(r, l)[0, 1]); sp.append(spearmanr(r[::7], l[::7])[0])
    print(f'{d}: Pearson mean {np.mean(c):.3f} min {min(c):.3f} | Spearman mean {np.mean(sp):.3f} min {min(sp):.3f}  ({n} images)')
