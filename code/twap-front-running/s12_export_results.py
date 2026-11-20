"""Export the small result tables published in results/ (no wallet address, no TWAP id).
    python s12_export_results.py   -> results/signals.csv, results/daily.csv, results/monthly.csv
"""
import pandas as pd
from common import HERE, WORK
from s08d_impact import frozen_curves
from s08h_wake_v1 import B, K0, S0, in_sample, simulate
from s09b_oos import DAYS as OOS, oos_paths

OUT = HERE / "results"; OUT.mkdir(exist_ok=True)

if __name__ == "__main__":
    L = []
    for period, R in (("in sample", in_sample()), ("out of sample", oos_paths())):
        R = R[R["px entry"].notna()]
        L.append(pd.DataFrame({"period": period, "day": R.day, "coin": R.coin, "side": R.side.astype(int), "t1_ms": R.t1.astype("int64"),
                               "first_slice_usd": R.usd1.round(2), "hl_weight_pct": R["HL weight %"].round(3), "global_weight_pct": R["widened weight %"].round(3),
                               "cancelled": R.cancelled.fillna(False).astype(bool), "announcement_found": R.found.astype(bool),
                               "net_bp_exit_A": R["A2 +2000"].round(3), "net_bp_exit_B": R["B +2000"].round(3)}))
    S = pd.concat(L).sort_values(["period", "t1_ms"]); S.to_csv(OUT / "signals.csv", index=False)
    Zs, edges = frozen_curves(); D = []
    for period, R, days in (("in sample", in_sample(), None), ("out of sample", oos_paths(), OOS)):
        R = R[R["px entry"].notna() & R["A2 +2000"].notna()]; days = days or sorted(R.day.unique())
        P = simulate(R, 0.01 * 25_000 / B, S0, K0, "A2 +2000", Zs, edges, per_coin=2)
        D.append(pd.DataFrame({"period": period, "day": days, "edge_bp_daily_mean": R.groupby("day")["A2 +2000"].mean().reindex(days).round(3).to_numpy(),
                               "signals": R.groupby("day").size().reindex(days, fill_value=0).to_numpy(),
                               "wake_v2_25k_1pct_usd": P[P.taken].groupby("day")["pnl $"].sum().reindex(days, fill_value=0).round(2).to_numpy()}))
    pd.concat(D).to_csv(OUT / "daily.csv", index=False)
    M = pd.read_parquet(WORK / "decay.parquet"); g = M.groupby("month")
    pd.DataFrame({"signals": g.size(), "net_edge_bp": g["net bp"].mean().round(2), "total_move_bp": g["total bp"].mean().round(2),
                  "taken_before_entry_bp": g["before me bp"].mean().round(2), "followed_within_2s_pct": (100 * g["followed within 2 s"].mean()).round(2),
                  "activity_30min_median_bp": g["activity 30 min bp"].median().round(2), "cancelled_pct": (100 * g["cancelled"].mean()).round(2)}).to_csv(OUT / "monthly.csv")
    print(len(S), "signals;", sum(len(x) for x in D), "days")
