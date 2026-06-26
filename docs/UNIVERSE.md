# Ticker universe construction

The trading universe is the set of symbols we stream and compute features for. It is
rebuilt per trading day so that any historical day has a **point-in-time** membership
(a symbol delisted later still appears on the days it was active — no survivorship bias).

Implementation: `quantzero/universe.py`. Build it with:

```bash
make universe                              # today, top 1000 by ADV$
python -m quantzero.universe build --date 2026-06-24 --max-symbols 1500
```

## The process

### 1. Pull all active US equities
We ask Alpaca's Trading API for every active US-equity asset:

```python
TradingClient(...).get_all_assets(
    GetAssetsRequest(status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY)
)
```

This returns roughly **7,000–7,500** assets (the count drifts as listings change).

### 2. Structural filters (non-price)
Keep only real, streamable common stock. A symbol is dropped unless it is:

| Filter | Rule | Why |
|--------|------|-----|
| Tradable | `asset.tradable` is true | Alpaca won't route others |
| Major exchange | exchange ∈ {NASDAQ, NYSE, AMEX, ARCA, BATS} | OTC/pink has no reliable SIP depth |
| Not fractional-only | `"/"` not in symbol | excludes fractional/fund-like identifiers |
| Not ETF/leveraged | name fails the `is_etf_like` regex | we want single-name equities, not funds |

`is_etf_like` matches names containing ETF/ETN/ProShares/Direxion/iShares/SPDR/Invesco/
VanEck/Global X/WisdomTree/UltraPro/Ultra-Short/Leveraged/Inverse/Index Fund/Income Fund/
Exchange-Traded/Bull/Bear/1-3X (case-insensitive). It is a heuristic — tune as needed.

After this step we typically have a few thousand candidate single-name equities.

### 3. Liquidity stats (price + ADV$)
For each candidate we fetch the last `LOOKBACK_DAYS = 20` **daily** bars (batched 200
symbols per request) and compute **average daily dollar volume**:

```
adv_dollar = mean(close * volume over the last 20 daily bars)
price       = most recent daily close
```

### 4. Threshold + rank
Keep names with `price >= $5` and `adv_dollar >= $10M`, then sort by **ADV$ descending**
(ties broken alphabetically by symbol, for determinism) and take the top `max_symbols`
(default **1000**). Raise `--max-symbols` to seed a larger set.

### 5. Persist (point-in-time)
The selection is written to parquet, partitioned by date:

```
<store>/universe/date=<YYYY-MM-DD>/universe.parquet
columns: trade_date, rank, symbol, price, adv_dollar
```

Each day is an independent snapshot, which is what gives backtests a point-in-time
universe.

## Tuning knobs (constants in `universe.py`)

| Constant | Default | Meaning |
|----------|---------|---------|
| `KEEP_EXCHANGES` | NASDAQ/NYSE/AMEX/ARCA/BATS | allowed listing venues |
| `MIN_PRICE` | 5.0 | minimum last close |
| `MIN_ADV_DOLLAR` | 10_000_000 | minimum average daily dollar volume |
| `DEFAULT_MAX_SYMBOLS` | 1000 | size of the selected universe |
| `LOOKBACK_DAYS` | 20 | daily bars used for ADV$ |
| `FETCH_CHUNK` | 200 | symbols per historical request |

## Open items / future work
- Cache the daily-bar fetch (the ADV pass is the slow part; ~thousands of symbols).
- Add a median-spread liquidity gate once we capture quotes at scale.
- Drive the daily rebuild from a scheduler (cron at pre-market), like quant-fp's loop.
