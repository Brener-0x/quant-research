"""Section 3. Inventory of Hyperliquid TWAPs over the 150 in-sample days.

Every fill carrying a twapId is kept; a TWAP is a (wallet, twapId) pair.
    python s03_inventory.py   -> work/twap_fills/<day>.parquet, work/twap_agg/<day>.parquet, work/twaps.parquet
"""
from multiprocessing import Pool
import numpy as np, pandas as pd
from common import DAYS, WORK, coin_reference, fills, show, work

TW, AGG = work("twap_fills"), work("twap_agg")


def one_day(day):
    f = TW / f"{day}.parquet"
    if f.exists() and (AGG / f"{day}.parquet").exists():
        return day, -1
    L = []
    for u, e in fills(day):
        if e.get("twapId") is not None:
            L.append((u, e["twapId"], e["coin"], e["time"], float(e["px"]), float(e["sz"]), 1 if e["side"] == "B" else -1,
                      bool(e["crossed"]), e.get("dir", ""), float(e["fee"]), e["feeToken"], e["oid"]))
    D = pd.DataFrame(L, columns=["user", "twapId", "coin", "t", "px", "sz", "side", "taker", "dir", "fee", "feeToken", "oid"])
    D["usd"] = D.px * D.sz; D["fee_usd"] = np.where(D.feeToken == "USDC", D.fee, np.nan); D = D.drop(columns=["fee", "feeToken"])
    D.to_parquet(f, index=False)
    D.assign(one=1).groupby(["user", "twapId"]).agg(coin=("coin", "first"), coins=("coin", lambda x: "|".join(sorted(set(x)))), start=("t", "min"),
        end=("t", "max"), slices=("t", "nunique"), usd=("usd", "sum"), side=("side", "sum"), taker=("taker", "sum"), n=("one", "sum"),
        dir=("dir", "first"), fee=("fee_usd", "sum")).reset_index().to_parquet(AGG / f"{day}.parquet", index=False)
    return day, len(D)


def classify(coin, U):
    base = "HIP-3" if ":" in coin else ("spot" if coin.startswith("@") or "/" in coin else "perp")
    ref = U.get(coin)
    return base, ("unclassified" if ref is None else ("external reference" if ref.startswith("external") else ref))


if __name__ == "__main__":
    with Pool(6) as pool:
        for d, n in pool.imap_unordered(one_day, DAYS):
            print(d, n, flush=True)
    A = pd.concat([pd.read_parquet(AGG / f"{d}.parquet") for d in DAYS], ignore_index=True)
    T = A.groupby(["user", "twapId"]).agg(coin=("coin", "first"), coins=("coins", lambda x: "|".join(sorted(set("|".join(x).split("|"))))),
                                          start=("start", "min"), end=("end", "max"), slices=("slices", "sum"), usd=("usd", "sum"),
                                          side=("side", "sum"), taker=("taker", "sum"), n=("n", "sum"), dir=("dir", "first"), fee=("fee", "sum")).reset_index()
    T["n_coins"] = T.coins.str.count(r"\|") + 1; T["side"] = T.side / T.n; T["taker"] = T.taker / T.n
    T["minutes"] = (T.end - T.start) / 60_000
    T["net_side"] = np.sign(T.side); T["mixed_side"] = T.side.abs() < 1
    T["month"] = pd.to_datetime(T.start, unit="ms").dt.strftime("%Y%m")
    U = coin_reference().set_index("coin").reference.to_dict()
    T[["type", "reference"]] = pd.DataFrame([classify(c, U) for c in T.coin], index=T.index)
    T["opens"] = T["dir"].str.startswith("Open")
    # share of the coin's volume between the first and the last fill (5-second bars)
    T["day0"] = pd.to_datetime(T.start, unit="ms").dt.strftime("%Y%m%d"); T["day1"] = pd.to_datetime(T.end, unit="ms").dt.strftime("%Y%m%d")
    vol = np.full(len(T), np.nan)
    for d in DAYS:
        idx = np.flatnonzero((T.day0 <= d) & (T.day1 >= d))
        if len(idx) == 0:
            continue
        B = pd.read_parquet(WORK / "bars5s" / f"{d}.parquet", columns=["coin", "t5", "notional"]); B["coin"] = B.coin.astype(str)
        G = {c: (g.t5.to_numpy(), np.concatenate([[0], np.cumsum(g.notional.to_numpy())])) for c, g in B.sort_values("t5").groupby("coin")}
        for i in idx:
            g = G.get(T.coin.iat[i])
            if g is None:
                continue
            a, b = np.searchsorted(g[0], T.start.iat[i] - 5000, side="right"), np.searchsorted(g[0], T.end.iat[i], side="right")
            vol[i] = np.nan_to_num(vol[i]) + g[1][b] - g[1][a]
    T["coin volume during"] = vol
    T["volume share %"] = 100 * T.usd / T["coin volume during"]
    T.to_parquet(WORK / "twaps.parquet", index=False)
    show(); q = [.1, .25, .5, .75, .9, .99]
    print(f"{int(T.n.sum()):,} TWAP fills, {len(T):,} TWAPs, {T.user.nunique():,} wallets, {T.coin.nunique()} coins; "
          f"notional {T.usd.sum() / 1e9:.2f} billion dollars; taker share of fills {(T.taker * T.n).sum() / T.n.sum():.1%}")
    print("\nQuantiles:\n", T[["usd", "minutes", "slices", "volume share %"]].quantile(q).to_string())
    print("\nBy type and reference:\n", T.groupby(["type", "reference"]).agg(TWAPs=("usd", "size"), notional=("usd", "sum"), median=("usd", "median"),
          share_median=("volume share %", "median")).to_string())
    print("\nOpening a position:", T.opens.mean().round(3), "of TWAPs")
    G = T.groupby("coin").agg(TWAPs=("usd", "size"), notional=("usd", "sum"), share_median=("volume share %", "median")).sort_values("notional", ascending=False)
    print("\nTop coins by notional:\n", G.head(10).to_string())
    for s in (1e5, 1e6):
        x = T[T.usd >= s]; print(f"TWAPs >= {s:,.0f} dollars: {len(x):,} ({x.usd.sum() / T.usd.sum():.0%} of notional), median volume share {x['volume share %'].median():.2f} %")
    W = T.groupby("user").usd.sum().sort_values(ascending=False)
    print(f"20 largest wallets: {W.head(20).sum() / W.sum():.0%} of notional")
