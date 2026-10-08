import sys, warnings, itertools, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
import cb_ele as c
def run(a):
    thr, w, dec, seed = a
    cfg = replace(c.CONFIG, seed=seed, rounds=1300, verbose_timing=False, follower_informed_fraction=0.3,
                  leader_recognition="observe", recognition_threshold=thr, recognition_slow_weight=w, recognition_decay=dec)
    t = c.run_simulation(cfg)[2]
    return dict(thr=thr, w=w, decay=dec, seed=seed, prec=np.nanmean(t.recognition_precision), rec=np.nanmean(t.recognition_recall))
if __name__ == "__main__":
    tasks = [(thr, w, dec, s) for thr in (2.0, 5.0) for w in (0.5, 1.0, 2.0) for dec in (0.995, 0.999) for s in range(1000, 1004)]
    with ProcessPoolExecutor(4) as p:
        df = pd.DataFrame(list(p.map(run, tasks)))
    g = df.groupby(["thr", "w", "decay"])[["prec", "rec"]].mean()
    g["f1"] = 2 * g.prec * g.rec / (g.prec + g.rec)
    print(g.sort_values("f1", ascending=False).round(3).to_string())
