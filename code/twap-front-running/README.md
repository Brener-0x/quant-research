# Front-Running TWAPs on Hyperliquid: replication code

Code for [Research note No. 01](../../papers/twap-front-running/). One script per step of the note, in the order of its
sections; each script states its protocol (and, for pre-registered tests, its criterion) at the top and prints the
numbers quoted in the note. Figures in the note are drawn from these outputs.

## Data

All inputs are public.

| Source | What | Used by |
|---|---|---|
| Hyperliquid node fills (`node_fills_by_block`, hourly `.lz4` files, one folder per day) | every fill with both wallets, side, price, size, fee, hash, `twapId` | `s00`, `s03`, `s04a`, `s04c`, `s05b`, `s08e`, `s09a`, `s11a` |
| `https://api.hyperliquid.xyz/info`, type `twapHistory` | TWAP announcements: size, duration, send time, final status | `s06c`, `s09b` |
| `data.binance.vision`, monthly 5-minute klines (USDT perps and spot) | global volume | `s07b`, `s07c`, `s09b` |

Frozen files in `data/`:

- `coin_reference.csv`: for each Hyperliquid coin, its market type and whether it has an external reference market
  (Binance or Bybit USDT perp), with the Binance symbol.
- `bar_coins.csv`: the coins covered by the in-sample 5-second volume bars. This list was inherited from an earlier study
  of wallet activity; with `coin_reference.csv` it defines the 170 in-sample perps with an external reference.
- `universe_oos.csv`: the out-of-sample universe (every perp with an external reference, frozen before reading).
- `max_leverage.csv`: maximum leverage per perp, used for the isolated-margin bound.

## Running

```
pip install numpy pandas pyarrow lz4
export HL_FILLS=/path/to/node_fills_by_block     # one sub-folder per day, YYYYMMDD
export TWAP_WORK=/path/to/work                   # derived tables (about 30 GB)
```

Steps that read raw fills are the slow ones (`s00`, `s03`, `s04a`, `s04c`, `s05b`, `s09a`, `s11a`). Downloads are
separate: `python s06c_announcements.py --download`, `python s07b_global_weight.py --download`,
`python s07c_reasons.py --download`, `python s09b_oos.py --download`, `python s09b_oos.py --api`.
Then run the scripts in order:

| Section | Script | What it measures |
|---|---|---|
| 2 | `s00_volume_bars.py` | 5-second taker volume bars (weight of a TWAP) |
| 3 | `s03_inventory.py` | inventory of TWAPs |
| 4 | `s04a_drift.py` | naive drift after the first slice (size known after the fact) |
| 4 | `s04b_controls.py` | drift without look-ahead, random and momentum controls |
| 4 | `s04c_small_markets.py` | markets without an external reference, at prices paid |
| 5 | `s05a_weight.py` | weight at entry against matched momentum |
| 5 | `s05b_heavy_executable.py` | heavy TWAPs at prices paid; base fee schedule by market type |
| 6 | `s06a_price_path.py` | path in units of the TWAP's duration |
| 6 | `s06b_maker_exit.py` | maker exit in the last quarter against taker exit at the end |
| 6 | `s06c_announcements.py` | announcements from the public API |
| 6 | `s06d_announcement_entry.py` | entry at the announcement; other takers before the first slice |
| 7 | `s07a_wallet_frequency.py` | the wallet's TWAP count over 7 days |
| 7 | `s07b_global_weight.py` | weight against global volume |
| 7 | `s07c_reasons.py` | reasons written in advance, selection and verification months |
| 7 | `s07d_cumulative_pressure.py` | cumulative pressure and herd; writes the core set |
| 7 | `s07e_end_without_announcement.py` | detecting the end from the slices alone |
| 8 | `s08a_entries.py` | entries E1, E2, E3 |
| 8 | `s08b_e2.py`, `s08c_exit_a.py` | E2 and exit A without look-ahead |
| 8 | `s08d_impact.py` | own impact from imposed-size slices; sizing |
| 8 | `s08e_null_hash.py`, `s08f_capacity.py` | slices in the public feed; signals per day |
| 8 | `s08g_wake_v0.py`, `s08h_wake_v1.py` | the frozen model, without and with protections |
| 9 | `s09a_oos_read.py`, `s09b_oos.py` | the single read of the five unseen months and the verdict |
| 9 | `s09c_robustness.py` | Newey-West and block bootstrap of the headline t statistics |
| 10 | `s10a_decay.py` | four explanations of the decay, month by month |
| 10 | `s10b_cross_section.py` | is the decay located where the followers are? |
| 11 | `s11a_counterparties.py`, `s11b_makers.py` | makers facing TWAP slices |
| 11 | `s11c_followers.py` | what the followers earn |
| | `s12_export_results.py` | the tables in `results/` |

## Results

`results/` holds the outputs most numbers rest on, without wallet addresses or TWAP ids:

- `signals.csv`: every E2 signal of the ten months (12,076 rows): day, coin, side, first slice, weights, fate of the
  TWAP, and the net result in bp (fees included, zero size) for exits A and B.
- `daily.csv`: per day, the mean edge, the number of signals, and the profit of the second version with 25,000 dollars
  at 1 % risk. The t statistics, Newey-West and bootstrap of Section 9 can be recomputed from it.
- `monthly.csv`: the month-by-month table of Section 10.

## Notes

- Inference is by day: a t statistic is the mean of daily means over its standard error.
- My own impact is modelled from TWAP slices (taker orders of imposed size), not observed: no historical order book
  exists for these months.
- No wallet address appears in the code or in its printed output beyond what the public data contains.

Code released under the MIT licence; the note under CC BY 4.0.
