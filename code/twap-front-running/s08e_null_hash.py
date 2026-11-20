"""Section 8. Can a TWAP slice be recognised in the public trade feed (which has no twapId)?

The `trades` websocket gives coin, side, price, size, hash and both wallets, but not the twapId. On raw fills
(in-sample months only, one day per month, all hours): null hash (0x000...0) against twapId present, and liquidations.
    python s08e_null_hash.py
"""
import collections
from common import fills

DAYS = ["20251015", "20251115", "20260215", "20260415", "20260515"]


def null(h):
    return set(h[2:]) == {"0"}


if __name__ == "__main__":
    total = collections.Counter()
    for d in DAYS:
        c = collections.Counter()
        for _, e in fills(d):
            c[("null" if null(e["hash"]) else "hash", "twap" if e.get("twapId") is not None else "-", "liq" if e.get("liquidation") else "-")] += 1
        total.update(c)
        n = sum(c.values()); z = sum(v for k, v in c.items() if k[0] == "null"); tw = sum(v for k, v in c.items() if k[1] == "twap")
        ztw = c[("null", "twap", "-")] + c[("null", "twap", "liq")]
        print(f"{d}: {n:,} fills; null hash {z:,} ({z / n:.1%}); with twapId {tw:,}; twapId among null hashes {ztw / max(z, 1):.2%}; "
              f"null hash among twapIds {ztw / max(tw, 1):.2%}")
    print("\nTotal (hash, twap, liquidation): fills")
    for k, v in sorted(total.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v:,}")
