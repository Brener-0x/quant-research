"""Section 8. Wake, second version: the first version plus protections set by reasoning BEFORE the computation
(no optimisation), then the capital: "with C dollars, at low, medium or high risk, what does it do?". In sample.

Protections:
  - catastrophe stop at -S = -5 % of the entry price: triggered at the first trade beyond, executed at the next opposite
    taker order (the price gap is counted); S = 3 and 10 % are measured to check the plateau, not to choose;
  - isolated margin: liquidation at -B = -30 %; a position loses at most its margin M = B + 1 / (2 x max leverage);
  - one (or two) positions per coin; K = 5 open positions at most; K = 3 and 8 measured;
  - at most 10 entries per 30 minutes; circuit breaker: after the 3rd stop or liquidation in 24 h, no entry for 24 h.
Size: risk r of capital C lost if the position reaches the hard bound (-30 %): N = r x C / 0.30, capped at 3 x the
Hyperliquid volume per 30 s (s08d). r = 0.5 % (low), 1 % (medium), 2 % (high).
    python s08h_wake_v1.py   -> work/wake_paths.parquet
"""
import numpy as np, pandas as pd
from common import DATA, DAYS, TAKER_FEE, WORK, first_after, show
from s08a_entries import ENTRY_WIN, EXIT_WIN, day_after
from s08d_impact import frozen_curves, tercile_cost

STOPS = [0.03, 0.05, 0.10]; S0 = 0.05
B = 0.30
KS = [3, 5, 8]; K0 = 5
PACE, PACE_WIN = 10, 1_800_000
CB_N, CB_WIN = 3, 86_400_000
CAP = 3
CAPITALS = [10_000, 25_000, 50_000, 100_000]
RISKS = {"low (0.5 %)": 0.005, "medium (1 %)": 0.01, "high (2 %)": 0.02}
LEV = pd.read_csv(DATA / "max_leverage.csv").set_index("coin").max_leverage.to_dict()


def paths(K, days, orders="orders", out=WORK / "wake_paths.parquet"):
    """For each E2 position: entry price, and for each stop S the time and net result (fees included) if it triggers
    before the planned exit; loss capped by the isolated margin."""
    if out.exists():
        return pd.read_parquet(out)
    K = K.copy()
    K["t entry"] = K.t1 + 2000
    K["t exit"] = np.maximum(np.where(K["t exit A2"].notna(), K["t exit A2"], K.end + 35_000), K["t entry"]).astype(np.int64)
    L = []
    for d in days:
        Sd = K[K.day == d]
        if len(Sd) == 0:
            continue
        nx = (pd.Timestamp(d) + pd.Timedelta(days=1)).strftime("%Y%m%d")
        dd_ = [x for x in (d, nx) if (WORK / orders / f"{x}.parquet").exists()]
        Od = pd.concat([pd.read_parquet(WORK / orders / f"{x}.parquet", columns=["coin", "t", "side", "px"]) for x in dd_]); Od = Od[Od.coin.isin(set(Sd.coin))]
        for coin, G in Sd.groupby("coin"):
            Oc = Od[Od.coin == coin].sort_values("t"); ta, pa = Oc.t.to_numpy(), Oc.px.to_numpy()
            book = {s: (Oc[Oc.side == s].t.to_numpy(), Oc[Oc.side == s].px.to_numpy()) for s in (1, -1)}
            lev = LEV.get(coin, 3); M = B + 1 / (2 * lev)
            for r in G.to_dict("records"):
                sg, te0, ts = int(r["side"]), int(r["t1"]) + 2000, int(r["t exit"])
                te, pe = first_after(*book[sg], np.array([te0]), np.array([te0 + ENTRY_WIN])); te, pe = te[0], pe[0]
                res = {"user": r["user"], "twapId": r["twapId"], "max leverage": lev, "margin M": M, "px entry": pe}
                if not np.isnan(pe):
                    a, b = np.searchsorted(ta, te, side="right"), np.searchsorted(ta, ts, side="right")
                    mv = sg * np.log(pa[a:b] / pe)
                    res["worst excursion %"] = 100 * mv.min() if len(mv) else 0.0
                    for S in STOPS:
                        k = np.flatnonzero(mv <= np.log(1 - S))
                        if len(k):
                            tt = int(ta[a + k[0]]); _, pf = first_after(*book[-sg], np.array([tt]), np.array([tt + EXIT_WIN]))
                            g = sg * np.log(pf[0] / pe) * 1e4 if not np.isnan(pf[0]) else np.log(1 - S) * 1e4
                            res[f"stop {S:.0%} t"] = tt; res[f"stop {S:.0%} bp"] = max(g, -M * 1e4) - 2 * TAKER_FEE
                        else:
                            res[f"stop {S:.0%} t"] = np.nan; res[f"stop {S:.0%} bp"] = np.nan
                L.append(res)
    R = K.merge(pd.DataFrame(L), on=["user", "twapId"], how="left")
    R.to_parquet(out, index=False); return R


def simulate(R, N, S=S0, K=K0, exit_col="A2 +2000", Zs=None, edges=None, per_coin=1):
    """Replay the signals in order with the protections; returns the positions taken (size, result, outcome)."""
    R = R[R["px entry"].notna() & R[exit_col].notna()].sort_values("t entry")
    v30 = R.vol30.to_numpy().astype(float); n = np.minimum(N, CAP * v30)
    c, _ = tercile_cost(n / np.where(v30 > 0, v30, 1), v30, Zs, edges)
    st, sp = R[f"stop {S:.0%} t"].to_numpy(), R[f"stop {S:.0%} bp"].to_numpy()
    te, ts0, net0, coin, M = R["t entry"].to_numpy(), R["t exit"].to_numpy(), R[exit_col].to_numpy(), R.coin.to_numpy(), R["margin M"].to_numpy()
    stopped = ~np.isnan(st) & (st < ts0)
    ts = np.where(stopped, st, ts0); gross = np.where(stopped, sp, np.maximum(net0, -M * 1e4 - 2 * TAKER_FEE))
    open_, entries, alerts, pause = [], [], [], -1
    taken = np.zeros(len(R), bool); refused = np.full(len(R), "", dtype=object)
    for i in range(len(R)):
        t = te[i]; open_ = [(f, c_) for f, c_ in open_ if f > t]
        alerts = [a for a in alerts if a > t - CB_WIN]; entries = [e for e in entries if e > t - PACE_WIN]
        if t < pause:
            refused[i] = "circuit breaker"; continue
        if sum(c_ == coin[i] for _, c_ in open_) >= per_coin:
            refused[i] = "coin already open"; continue
        if len(open_) >= K:
            refused[i] = f"{K} positions open"; continue
        if len(entries) >= PACE:
            refused[i] = "pace"; continue
        taken[i] = True; open_.append((ts[i], coin[i])); entries.append(t)
        if stopped[i]:
            alerts.append(ts[i])
            if len([a for a in alerts if a <= ts[i]]) >= CB_N:
                pause = ts[i] + CB_WIN
    P = pd.DataFrame({"day": R.day.to_numpy(), "month": R.month.to_numpy(), "coin": coin, "t entry": te, "t exit": ts, "size $": n,
                      "margin $": n * M, "stopped": stopped, "net bp": gross - 2 * c, "taken": taken, "refused": refused})
    P["pnl $"] = np.where(P.taken, P["size $"] * P["net bp"] / 1e4, 0.0)
    return P


def summary(P, capital, days):
    p = P[P.taken]; d = p.groupby("day")["pnl $"].sum().reindex(days, fill_value=0)
    cum = capital + d.cumsum(); dd = (cum / cum.cummax() - 1).min()
    month = p.groupby("month")["pnl $"].sum(); per_month = month.sum() / (len(days) / 30)        # 30-day months
    return {"positions": len(p), "refused": int((~P.taken).sum()), "stopped": int(p.stopped.sum()), "total $": d.sum(), "% per month": 100 * per_month / capital,
            "worst month %": 100 * month.min() / capital, "best month %": 100 * month.max() / capital, "days > 0 %": 100 * (d > 0).mean(),
            "worst day %": 100 * d.min() / capital, "largest drawdown %": 100 * dd, "worst position %": 100 * p["pnl $"].min() / capital,
            "daily t": d.mean() / d.std() * np.sqrt(len(d))}


def in_sample():
    """Positions of the in-sample model, with their paths."""
    return paths(pd.read_parquet(WORK / "e2.parquet"), DAYS)


if __name__ == "__main__":
    R = in_sample(); Zs, edges = frozen_curves()
    days = sorted(R.day.unique())
    show()
    ok = R["px entry"].notna()
    print(f"Possible positions: {ok.sum():,}; worst excursion during the position (%), quantiles: " + str(R.loc[ok, "worst excursion %"].quantile([.001, .01, .05, .5]).round(2).to_dict()))
    C0, r0 = 25_000, 0.01; N0 = r0 * C0 / B
    rows = []
    for S in STOPS:
        for K in KS:
            b = summary(simulate(R, N0, S, K, "A2 +2000", Zs, edges), C0, days); rows.append({"stop": f"{S:.0%}", "K": K, **b})
    print(f"\n1. Plateau of the protections (capital {C0:,} dollars, 1 % risk: N = {N0:,.0f} dollars, one per coin):\n" + pd.DataFrame(rows).to_string(index=False))
    rows = []
    for Cap in (10_000, 25_000, 100_000):
        for name, r in RISKS.items():
            for pc in (1, 2):
                b = summary(simulate(R, r * Cap / B, S0, K0, "A2 +2000", Zs, edges, per_coin=pc), Cap, days)
                rows.append({"capital $": Cap, "risk": name, "per coin": pc, **{k: b[k] for k in ("positions", "total $", "% per month", "worst month %", "largest drawdown %", "daily t")}})
    print("\n2. One or two positions per coin (stop 5 %, 5 positions max, exit A):\n" + pd.DataFrame(rows).to_string(index=False))
