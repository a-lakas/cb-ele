"""Which observable behaviours separate leaders from followers?  Computes
per-drone, per-round features any neighbour could observe and their AUC
(leader vs follower).  Usage: python recognition_features.py OUT.csv"""
import sys, numpy as np, pandas as pd
sys.path.insert(0, ".")
from dataclasses import replace
from concurrent.futures import ProcessPoolExecutor
from scipy.stats import rankdata
import cb_ele as c

def ema(x, a):
    out = np.zeros_like(x); acc = np.zeros(x.shape[1])
    for t in range(len(x)):
        acc = a * acc + x[t]; out[t] = acc
    return out

def feats(seed, informed):
    cfg = replace(c.CONFIG, seed=seed, rounds=2000, verbose_timing=False,
                  follower_informed_fraction=informed)
    traj = c.run_simulation(cfg, return_trajectory=True)[-1]
    P, V, S = traj["position"], traj["velocity"], traj["state"]
    T, n, _ = P.shape
    F = {k: np.zeros((T, n)) for k in
         ["freeze", "retreat", "nonyield12", "yield12", "front", "persist", "rel_speed", "approach"]}
    for t in range(2, T):
        d = np.linalg.norm(P[t-1][:, None] - P[t-1][None], axis=2); np.fill_diagonal(d, np.inf)
        nb = d <= cfg.sensing_radius
        sp = np.linalg.norm(V[t], axis=1)
        contact = (d <= cfg.election_radius).any(1)
        F["freeze"][t] = contact & (sp < 0.05)
        F["retreat"][t] = contact & (sp >= 0.05)
        for j in range(n):
            ids = np.flatnonzero(nb[j])
            if not len(ids): continue
            close = np.flatnonzero(d[j] <= 12.0)
            if len(close):
                away = P[t-1][j] - P[t-1][close]
                away /= np.linalg.norm(away, axis=1, keepdims=True)
                radial = (V[t][j] @ away.T).mean()
                F["yield12"][t, j] = radial > 0.1
                F["nonyield12"][t, j] = radial <= 0.1
            mv = V[t-1][ids].mean(0); nm = np.linalg.norm(mv)
            if nm > 1e-6:
                F["front"][t, j] = (P[t-1][j] - P[t-1][ids].mean(0)) @ (mv / nm) / cfg.sensing_radius
            F["rel_speed"][t, j] = sp[j] - sp[ids].mean()
            # approaching neighbours: velocity toward neighbour centroid
            to_c = P[t-1][ids].mean(0) - P[t-1][j]; nc = np.linalg.norm(to_c)
            if nc > 1e-6 and sp[j] > 1e-6:
                F["approach"][t, j] = V[t][j] @ to_c / (nc * sp[j])
        a, b = V[t], V[t-1]
        na, nb_ = np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1)
        ok = (na > 1e-6) & (nb_ > 1e-6)
        F["persist"][t][ok] = np.sum(a[ok] * b[ok], 1) / (na[ok] * nb_[ok])
    lead, foll = S == c.LEADER, S == c.FOLLOWER
    rows = []
    for k, x in F.items():
        for tag, xx in (("raw", x), ("ema.95", ema(x, 0.95) * 0.05),
                        ("ema.995", ema(x, 0.995) * 0.005)):
            rows.append(dict(feature=k, smooth=tag, seed=seed, informed=informed,
                             lead_mean=xx[lead].mean(), foll_mean=xx[foll].mean(),
                             auc=auc(xx[lead], xx[foll])))
    return rows

def auc(pos, neg):
    r = rankdata(np.concatenate([pos, neg]))
    return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))

if __name__ == "__main__":
    tasks = [(s, f) for f in (0.0, 0.3) for s in range(6)]
    with ProcessPoolExecutor(4) as p:
        rows = [r for rs in p.map(feats, *zip(*tasks)) for r in rs]
    df = pd.DataFrame(rows)
    df.to_csv(sys.argv[1], index=False)
    print(df.groupby(["feature", "smooth"])[["auc", "lead_mean", "foll_mean"]].mean()
          .sort_values("auc", ascending=False).round(3).to_string())
