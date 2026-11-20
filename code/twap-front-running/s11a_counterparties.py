"""Section 11. Where does the money a TWAP pays go? Step 1: extraction. For every fill of a TWAP slice (taker, twapId),
the wallet on the other side (maker, same tid, same block), on two days per month over the ten months (the 10th and the
20th). No analysis here (step 2: s11b).
    python s11a_counterparties.py   -> work/counterparties/<day>.parquet
"""
import json, time
import pandas as pd
from common import IN_SAMPLE, OUT_OF_SAMPLE, RAW, STABLES, work

OUT = work("counterparties")
DAYS = [m + d for m in IN_SAMPLE + OUT_OF_SAMPLE for d in ("10", "20")]


def one_day(day):
    f = OUT / f"{day}.parquet"
    if f.exists():
        return day, -1
    import lz4.frame
    L = []
    for fic in sorted((RAW / day).glob("*.lz4"), key=lambda p: int(p.stem)):
        with lz4.frame.open(fic, "rt", encoding="utf-8") as fh:
            for line in fh:
                ev = json.loads(line)["events"]                        # one block
                tw = {e["tid"]: (u, e) for u, e in ev if e.get("twapId") is not None and e["crossed"]}
                if not tw:
                    continue
                for u, e in ev:
                    if not e["crossed"] and e["tid"] in tw:
                        ut, et = tw[e["tid"]]; px, sz = float(et["px"]), float(et["sz"])
                        fee = float(e["fee"]); fee = fee if e["feeToken"] in STABLES else fee * px
                        L.append((day, et["coin"], et["time"], e["tid"], ut, et["twapId"], 1 if et["side"] == "B" else -1, px, sz, px * sz, u, fee, e["oid"]))
    D = pd.DataFrame(L, columns=["day", "coin", "t", "tid", "twap_user", "twapId", "twap_side", "px", "sz", "usd", "maker", "maker_fee", "maker_oid"])
    D.to_parquet(f, index=False); return day, len(D)


if __name__ == "__main__":
    for d in DAYS:
        print("{}: {} TWAP fills matched".format(*one_day(d)), flush=True)
