"""Section 6. What do the TWAP announcements of the public API say (cancellations, announced weight, lead over the
first slice)? Free endpoint https://api.hyperliquid.xyz/info, type twapHistory. Only entries sent between 2025-10-01
and 2026-05-31 are kept (nothing later is written or shown).
    python s06c_announcements.py --download   -> work/api/<wallet>.json (resumable)
    python s06c_announcements.py              -> work/announcements.parquet
"""
import json, sys, time, urllib.request
import numpy as np, pandas as pd
from common import WORK, coin_type, show, t_day, work
from s05b_heavy_executable import universe

API = work("api")
START, END = 1759276800000, 1780271999999            # 2025-10-01 00:00 and 2026-05-31 23:59:59.999 UTC
PAUSE = 1.05                                          # about one request per second


def wallets():
    """Wallets of TWAPs of weight >= 5 % on referenced coins; those of heavy perps first."""
    X = universe(); X = X[X["weight1 %"] >= 5]
    perp = X.coin.map(coin_type) == "perp"
    first = X[perp & (X["weight1 %"] >= 20)].user.value_counts().index.tolist()
    return first + sorted(set(X.user) - set(first))


def sent(e):
    st = e.get("state", {}) if isinstance(e, dict) else {}
    t = st.get("timestamp")
    return t if t is not None else (e.get("time", 0) * 1000 if isinstance(e, dict) else 0)


def one(u, folder=API, keep=lambda e: START <= sent(e) <= END):
    """Download one wallet's twapHistory and keep only the entries of the study window."""
    f = folder / f"{u}.json"
    if f.exists():
        return 0
    r = None
    for k in range(6):
        try:
            req = urllib.request.Request("https://api.hyperliquid.xyz/info", data=json.dumps({"type": "twapHistory", "user": u}).encode(),
                                         headers={"Content-Type": "application/json"})
            r = json.loads(urllib.request.urlopen(req, timeout=30).read()); break
        except Exception:
            time.sleep(5 * (k + 1))
    if r is None:
        return 0
    kept = [e for e in r if keep(e)]
    f.write_text(json.dumps({"received": len(r), "kept": kept}), encoding="utf-8")
    time.sleep(PAUSE); return 1


def read_api(folder):
    """One row per TWAP: the most advanced entry (final status when present)."""
    L, capped = [], 0
    for f in folder.glob("*.json"):
        d = json.loads(f.read_text(encoding="utf-8")); capped += d["received"] >= 2000
        for e in d["kept"]:
            st = e.get("state", {}); s_ = e.get("status", {})
            L.append({"user": st.get("user", f.stem), "twapId": e.get("twapId"), "coin": st.get("coin"), "side": 1 if st.get("side") == "B" else -1,
                      "sz": float(st.get("sz", "nan")), "exec_sz": float(st.get("executedSz", "nan")), "exec_usd": float(st.get("executedNtl", "nan")),
                      "minutes": st.get("minutes"), "reduce": st.get("reduceOnly"), "randomize": st.get("randomize"), "trigger": st.get("trigger") is not None,
                      "t_sent": st.get("timestamp"), "status": s_.get("status") if isinstance(s_, dict) else str(s_)})
    A = pd.DataFrame(L)
    A["rank"] = A.status.map({"activated": 0, "finished": 2, "terminated": 2, "error": 2}).fillna(1)
    F = A.sort_values(["twapId", "rank"]).groupby("twapId").agg(user=("user", "first"), coin=("coin", "first"), sz=("sz", "first"), minutes=("minutes", "first"),
                                                               t_sent=("t_sent", "first"), status=("status", "last"), exec_sz=("exec_sz", "last"),
                                                               exec_usd=("exec_usd", "last"), trigger=("trigger", "first"), randomize=("randomize", "first")).reset_index()
    return F, capped


if __name__ == "__main__":
    if "--download" in sys.argv:
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(2) as ex:
            list(ex.map(one, wallets()))
        sys.exit()
    F, capped = read_api(API)
    X = universe().merge(pd.read_parquet(WORK / "twaps.parquet", columns=["user", "twapId", "end", "usd"]), on=["user", "twapId"])
    M = X.merge(F.drop(columns=["user", "coin"]), on="twapId", how="left")
    M["found"] = M.sz.notna(); M["month"] = M.day.str[:6]
    print(f"{capped} wallets at the 2,000-entry cap; coverage of TWAPs of weight >= 5 %: {M[M['weight1 %'] >= 5].found.mean():.1%}")
    T = M[M.found].copy()
    px1 = (T.usd / T.sz.where(T.exec_sz <= 0, T.exec_sz))
    T["announced usd"] = T.sz * px1
    T["announced weight %"] = 100 * T["announced usd"] / (2 * T.minutes) / T.vol30
    T["executed / announced"] = T.exec_sz / T.sz
    T["lead over 1st slice (s)"] = (T.t1 - T.t_sent) / 1000
    T["observed / announced duration"] = (T.end - T.t1) / 60_000 / T.minutes
    T["cancelled"] = T.status.isin(["terminated"]) | (T["executed / announced"] < 0.95)
    T.to_parquet(WORK / "announcements.parquet", index=False)
    show(); q = [.01, .1, .25, .5, .75, .9, .99]
    print(f"{len(T):,} TWAPs matched; final status {T.status.value_counts().to_dict()}")
    print("Lead of the announcement over the first slice (s):", T["lead over 1st slice (s)"].quantile(q).round(2).to_dict())
    print("Executed / announced:", T["executed / announced"].quantile(q).round(3).to_dict())
    print("\nBy weight class:\n", T.groupby("weight class", observed=True).agg(n=("twapId", "size"), cancelled=("cancelled", "mean"),
          lead_s=("lead over 1st slice (s)", "median"), minutes=("minutes", "median")).to_string())
    # Description after the fact (the final status is not known at entry): the s06b result by fate of the TWAP.
    G = pd.read_parquet(WORK / "maker_exit.parquet"); G = G[(G.get("control") != True) & (G.type == "perp") & (G["weight class"] == ">= 20 %") & (G.entered == True)]
    G = G.merge(T[["user", "twapId", "status", "cancelled"]], on=["user", "twapId"], how="left"); g = G[G.status.notna()]
    print("\nTaker exit at the end, perps >= 20 %, by status (after the fact):\n", g.groupby("status")["taker at end"].agg(["size", "mean", "median"]).to_string())
