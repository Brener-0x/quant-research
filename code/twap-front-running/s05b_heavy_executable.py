"""Section 5. Does following a heavy TWAP (weight >= 20 %) on a referenced coin pay at prices actually paid?

Pre-registered: TWAPs of s05a on coins with an external reference; weight = first slice alone / the coin's volume per
30 s over the previous hour. Entries: first and second slice + 1 s, at the best level of the first taker order on my
side within 60 s; exit at the first opposite taker order between + 30 and + 40 min; the coin's fee both ways.
Criterion (>= 20 %, each entry): net > 0, daily t >= 3, and > 0 without the coin that contributes most.
    python s05b_heavy_executable.py   -> work/orders/<day>.parquet (one row per taker order), work/heavy_executable.parquet
"""
import json
from array import array
from multiprocessing import Pool
import numpy as np, pandas as pd
from common import DAYS, STABLES, WORK, coin_type, fills, first_after, show, t_day, work

ORD = work("orders")
WEIGHTS = [0, 1, 5, 20, np.inf]; WEIGHT_NAMES = ["< 1 %", "1-5 %", "5-20 %", ">= 20 %"]
ENTRY_WIN, HOLD, EXIT_WIN = 60_000, 1_800_000, 600_000
rng = np.random.default_rng(39)


def universe():
    """TWAPs on coins with an external reference, with the weight of their first slice."""
    X = pd.read_parquet(WORK / "weight.parquet", columns=["user", "twapId", "coin", "day", "class", "t1", "t2", "usd1", "vol30"])
    X = X[X["class"] == "external reference"].copy()
    X["weight1 %"] = 100 * X.usd1 / X.vol30
    X["weight class"] = pd.cut(X["weight1 %"], WEIGHTS, labels=WEIGHT_NAMES, right=False)
    T = pd.read_parquet(WORK / "twaps.parquet", columns=["user", "twapId", "net_side"])
    return X.merge(T, on=["user", "twapId"])


def one_day(day):
    f = ORD / f"{day}.parquet"
    if f.exists():
        return day, -1
    CO = set(json.loads((WORK / "orders_coins.json").read_text()))
    us, cs, oid, tt, sd = [], [], array("q"), array("q"), array("b"); P, U, F = array("d"), array("d"), array("d")
    for u, e in fills(day):
        if e["crossed"] and e["coin"] in CO:
            px, sz, fee = float(e["px"]), float(e["sz"]), float(e["fee"])
            us.append(u); cs.append(e["coin"]); oid.append(e["oid"]); tt.append(e["time"]); sd.append(1 if e["side"] == "B" else -1)
            P.append(px); U.append(px * sz); F.append(fee if e["feeToken"] in STABLES else fee * px)
    D = pd.DataFrame({"user": pd.Categorical(us), "coin": pd.Categorical(cs), "oid": np.frombuffer(oid, dtype=np.int64), "t": np.frombuffer(tt, dtype=np.int64),
                      "side": np.frombuffer(sd, dtype=np.int8), "px": np.frombuffer(P), "usd": np.frombuffer(U), "fee": np.frombuffer(F)})
    O = D.groupby(["user", "coin", "oid"], sort=False, observed=True).agg(t=("t", "min"), side=("side", "first"), pmin=("px", "min"), pmax=("px", "max"),
                                                                         usd=("usd", "sum"), fee=("fee", "sum")).reset_index()
    O["px"] = np.where(O.side > 0, O.pmin, O.pmax)          # best level touched
    O.drop(columns=["pmin", "pmax"]).sort_values("t").to_parquet(f, index=False); return day, len(O)


def coin_fees():
    """Taker fee per coin: 90th percentile over a sample (at most 2,000 orders per coin and day)."""
    s = []
    for d in DAYS:
        O = pd.read_parquet(ORD / f"{d}.parquet", columns=["coin", "usd", "fee"]); O = O[O.usd > 0]
        s.append(O.sample(frac=1, random_state=39).groupby("coin", observed=True).head(2000))
    O = pd.concat(s); return (O.fee / O.usd * 1e4).groupby(O.coin.astype(str)).quantile(.9)


if __name__ == "__main__":
    X = universe()
    co = sorted(X[X["weight1 %"] >= 5].coin.unique()); (WORK / "orders_coins.json").write_text(json.dumps(co))
    print(f"{len(X):,} TWAPs on referenced coins; {len(co)} coins with at least one TWAP of weight >= 5 %", flush=True)
    with Pool(3) as pool:
        for d, n in pool.imap_unordered(one_day, DAYS):
            print(d, n, flush=True)
    X = X[X.coin.isin(co)].reset_index(drop=True)
    FEE = coin_fees()
    L = []
    for d in DAYS:
        Xd = X[X.day == d]
        if len(Xd) == 0:
            continue
        i_ = DAYS.index(d); Od = pd.read_parquet(ORD / f"{d}.parquet", columns=["coin", "t", "side", "px"])
        if i_ + 1 < len(DAYS):
            Od = pd.concat([Od, pd.read_parquet(ORD / f"{DAYS[i_ + 1]}.parquet", columns=["coin", "t", "side", "px"])])
        d0 = int(pd.Timestamp(d).value // 1_000_000)
        for coin, G in Xd.groupby("coin"):
            Oc = Od[Od.coin == coin].sort_values("t")
            book = {s: (Oc[Oc.side == s].t.to_numpy(), Oc[Oc.side == s].px.to_numpy()) for s in (1, -1)}
            for who in ("TWAP", "random"):
                R = G[["user", "coin", "day", "weight class"]].copy(); R["who"] = who
                if who == "TWAP":
                    sg = G.net_side.to_numpy().astype(int); entries = {"1st": G.t1.to_numpy().astype(np.int64) + 1000, "2nd": G.t2.to_numpy().astype(np.int64) + 1000}
                else:
                    sg = rng.choice([-1, 1], size=len(G)); ta = d0 + rng.integers(60_000, 86_400_000 - 3_600_000, size=len(G)); entries = {"1st": ta, "2nd": ta + 30_000}
                for name, te0 in entries.items():
                    pe, ps = np.full(len(G), np.nan), np.full(len(G), np.nan)
                    for s in (1, -1):
                        m = sg == s
                        if not m.any():
                            continue
                        te, pe[m] = first_after(*book[s], te0[m], te0[m] + ENTRY_WIN)
                        ok = ~np.isnan(te); tes = np.where(ok, te, 0).astype(np.int64)
                        _, x = first_after(*book[-s], tes + HOLD, tes + HOLD + EXIT_WIN); ps[m] = np.where(ok, x, np.nan)
                    R[f"gross {name}"] = sg * np.log(ps / pe) * 1e4
                    R[f"net {name}"] = R[f"gross {name}"] - 2 * FEE.get(coin, np.nan)
                    R[f"entered {name}"] = ~np.isnan(pe)
                R["fees"] = 2 * FEE.get(coin, np.nan)
                L.append(R)
    M = pd.concat(L, ignore_index=True).replace([np.inf, -np.inf], np.nan)
    for c in [c for c in M.columns if c.startswith(("gross", "net"))]:
        M[c] = M[c].clip(-2000, 2000)
    M.to_parquet(WORK / "heavy_executable.parquet", index=False)
    show()
    for e in ("1st", "2nd"):
        g = M[(M.who == "TWAP") & (M["weight class"] == ">= 20 %")]; x = g[g[f"net {e}"].notna()]
        pc = x.groupby("coin").agg(n=(f"net {e}", "size"), net=(f"net {e}", "mean"), total=(f"net {e}", "sum"))
        top = pc.total.idxmax(); without = x[x.coin != top]
        print(f"Entry {e} slice + 1 s, weight >= 20 %: gross {x[f'gross {e}'].mean():.2f}, fees {x.fees.mean():.2f}, net {x[f'net {e}'].mean():.2f} bp "
              f"(t {t_day(x, f'net {e}'):.2f}); without {top} {without[f'net {e}'].mean():.2f}")
        ok = x[f"net {e}"].mean() > 0 and t_day(x, f"net {e}") >= 3 and without[f"net {e}"].mean() > 0
        print("Criterion: " + ("passed" if ok else "failed"))
    # Exploratory, after reading the result: the protocol's fee percentile includes builder fees I would not pay.
    # At the base schedule (perps 4.5 bp, spot 7 bp per side; HIP-3 keeps its percentile), by market type.
    x = M[(M.who == "TWAP") & (M["weight class"] == ">= 20 %") & M["net 1st"].notna()].copy()
    x["type"] = x.coin.map(coin_type)
    x["base fees"] = np.where(x.type == "perp", 9.0, np.where(x.type == "spot", 14.0, x.fees))
    x["net base"] = x["gross 1st"] - x["base fees"]
    print("\nBase schedule, entry 1st slice, by type:", x.groupby("type")["net base"].agg(["size", "mean"]).round(2).to_dict("index"))
    p = x[x.type == "perp"]
    print(f"Perps, weight >= 20 %: n {len(p)}, {p.coin.nunique()} coins, net {p['net base'].mean():.2f} bp (t {t_day(p, 'net base'):.2f}); "
          f"by month {p.assign(m=p.day.str[:6]).groupby('m')['net base'].mean().round(2).to_dict()}")
