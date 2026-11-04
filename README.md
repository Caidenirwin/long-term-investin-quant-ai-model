# Swing-Funded Dividend Portfolio (CAD)

This project is a beginner-friendly Python backtester for a two-part strategy:

1. **Swing sleeve**: breakout trades in liquid ETFs and large-cap stocks, plus a daily scanner (`scan.py`) that researches today's setups.
2. **Dividend core**: long-term dividend stocks spread across 8 industries. The swing sleeve's realized profits are swept into it, and its dividends are reinvested.

It uses free historical data from `yfinance`, converts U.S.-listed holdings into Canadian dollars, includes fees, slippage, and U.S. dividend withholding tax, and writes CSV reports to `outputs/`.

This is research software only. It does not place trades and does not connect to a brokerage.

## Your Plan: C$1,000 Start + Monthly Deposits

`starting_cash = 1000` and `[contributions] monthly = 200` in settings. Each deposit is split like the starting cash: 25% to the swing sleeve, 75% to dividend stocks.

Small accounts get realistic handling:
- **Whole shares only** (`whole_shares = true`).
- **Fewer dividend stocks while small.** Each holding is at least `min_position_value` (C$200), one per industry first. More are added as deposits grow the account, and stocks whose single share costs more than a position are skipped.
- **Fewer swing positions while small.** Each is at least `swing.min_trade_value` (C$200), and an unaffordable signal is skipped in favour of the next one.
- **Fees:** `transaction_fee = 1.00` (about Interactive Brokers' minimum). At C$4.95, building a C$1,000 portfolio would cost about 12% up front.

Returns are **time-weighted**, the standard way to measure performance without counting your deposits as gains. The comparison is **the same deposits invested in XIC**.

Backtest since 2012 (C$1,000 + C$200/month):

| | |
|---|---|
| Deposited | C$36,400 |
| Worth now | **C$95,653** |
| Same deposits in XIC | C$99,151 |
| Worst drop | −30% (XIC −37%) |
| Dividend income | C$2,980/yr |

`audit_backtest.py`, `optimize.py`, `walk_forward.py` and `test_new_companies.py` judge the **rules** on a fixed C$100k account with no deposits (`research_starting_cash`). That keeps their results comparable over time. `main.py`, the dashboard and the forward test use your real plan.

**Forward test with deposits:** `live_trader.py` adds the monthly deposit automatically in `sim` mode. With a real broker, add the money there, then run `python live_trader.py --deposit 200` so the book splits it between the sleeves.

## How Money Flows

```text
starting cash ──┬── 25% ──> SWING SLEEVE ── realized profit above its starting capital ──┐
                │                                                                       │
                └── 75% ──> DIVIDEND CORE <─────────────── profit sweep ─────────────────┘
                              │      ▲
                              └──────┘  dividends reinvested (DRIP)
```

- Swing losses stay inside the swing sleeve. Nothing is swept until the sleeve is back above its starting capital, so only **net** profit moves.
- Swept cash is invested in the core holdings that are furthest below their target weight, which keeps the industries balanced.
- Idle cash in either sleeve earns `cash_interest_rate`, like a high-interest savings ETF.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

Run the backtest (downloads fresh data):

```bash
python main.py
python main.py --cached     # reuse data/ CSVs instead of downloading
```

Open the dashboard (easiest way to use everything):

```bash
python dashboard.py
```

It opens in your browser at http://127.0.0.1:8050 and shows:
- Your portfolio's value over time, compared with the TSX.
- Dividend income, swing profits, and today's trade ideas, which you can filter by industry and click for details.
- An industry breakdown and your dividend holdings.

The **Run scan** and **Rerun backtest** buttons do the work for you. The dashboard runs only on your computer.

Find today's swing trades:

```bash
python scan.py                          # whole S&P 500 + TSX 60 plus your list (~1 minute)
python scan.py --no-discover            # only your configured list (fast)
python scan.py --tickers SHOP.TO NVDA   # research specific tickers
```

Test the rules on companies they were never tuned on:

```bash
python test_new_companies.py                              # built-in holdout lists
python test_new_companies.py --swing-tickers V MA SHOP.TO # your own list
```

Check that the backtest is trustworthy:

```bash
python audit_backtest.py
```

It runs five kinds of checks:
- **Ledger:** cash and share counts rebuilt from the trade log must match the engine to the cent.
- **Look-ahead:** cutting the data off early must not change a single earlier decision.
- **Data quality:** no bad price ticks or broken split adjustments.
- **Swing edge:** breakout trades compared with random trades.
- **Robustness:** other start years, double costs, slower execution, and FX fees.

Test whether re-tuning the settings would really beat the TSX, without using future data:

```bash
python walk_forward.py
```

For each year from 2015 on, it picks the best of 65 setting combinations using only the years before. Then it records how that pick did in the year itself, which it had never seen. It compares three things: that honest track record, keeping your current settings fixed, and the "best in hindsight" setting.

The result was:

| Approach | Per year | vs TSX |
|---|---|---|
| Re-tuning every year | 11.8% | +1.0% |
| **Current settings, fixed** | **12.1%** | **+1.3%** |
| Best in hindsight | 13.2% | +2.4% (only exists looking back) |

**Re-tuning didn't help**, so the current settings are kept. Edit `CHOICES` at the top of the file to test new ideas the same way before adopting them.

Run the optimizer:

```bash
python optimize.py --focus swing --max-runs 600   # tune the swing engine
python optimize.py --focus core                   # tune allocation and the dividend core
```

## Forward Testing and Live Trading (`live_trader.py`)

Run it **once per trading day**. It uses the same rule code as the backtest. In a two-year day-by-day replay it took the same 42 of 42 swing trades and finished within 0.3% of the backtest (whole shares account for the difference). `audit_backtest.py` re-checks this.

```bash
python live_trader.py                           # forward test: simulated fills at real closing prices
python live_trader.py --status                  # what it holds and what's waiting to fill
python live_trader.py --dry-run                 # show today's plan without changing anything
python live_trader.py --mode ibkr-paper         # Interactive Brokers paper account
python live_trader.py --mode ibkr-live --live   # real money (also needs allow_live = true)
```

**What each run does:**
1. Records fills from the last run's orders.
2. Credits dividends.
3. Sweeps net swing profit into the dividend core.
4. Plans new orders from the last completed close.
5. Runs the safety checks.
6. Sends market-on-close orders, or simulates them.

Orders fill at the next session's close, which matches the backtest's timing.

**Recommended path:**
1. **Forward test (`mode = "sim"`)** for at least 1–3 months. Watch the *Forward test* panel on the dashboard.
2. **IBKR paper.** Open an Interactive Brokers account and install **IB Gateway**, logged into the *paper* account. Then run `python -m pip install ib_async` and `python live_trader.py --mode ibkr-paper`. Run it at about **3:00 pm Toronto time**, because market-on-close orders must arrive before the 3:40 pm cutoff.
3. **Live**, only after paper matches expectations. Set `allow_live = true` under `[live]` and pass `--live`.

**Safety locks:**
- **Paper/live account check:** paper mode refuses to connect to a live account, and live mode refuses a paper account. IBKR paper accounts start with "D".
- **Size limits:** `max_order_value` and `max_orders_per_run` cap individual orders, and `max_run_turnover` caps the total traded in one run.
- **Pause switch:** create an empty file `live/PAUSE` to stop all new orders instantly. Delete it to resume.
- **Book/broker check:** if your broker's positions don't match the book (for example after a manual trade), nothing is traded until you fix it or run `--resync`.
- **Trading-hours check:** orders are only sent between 9:30 am and 3:35 pm on weekdays. A blocked run is retried on the next run, not skipped.
- **No duplicates:** running twice in a day never sends duplicate orders.

**Scheduling:** `run_daily.bat` runs the trader and logs to `live/run_log.txt`. To run the forward test at 5 pm on weekdays:

```bash
schtasks /Create /TN "LongTermQuant" /TR "\"%CD%\run_daily.bat\"" /SC WEEKLY /D MON,TUE,WED,THU,FRI /ST 17:00
```

For IBKR, use `/ST 15:00` instead.

**Files** (in `live/`):
- `book_<mode>.json`: holdings by sleeve
- `journal_<mode>.csv`: daily value
- `orders_<mode>.csv`
- `swing_trades_<mode>.csv`

## Swing Sleeve Rules

All swing rules live in [src/swing.py](src/swing.py) and are shared by the backtest and the scanner, so the scanner shows exactly what the backtest trades. Two setups are available in `swing.setups`:

**breakout** (the default)
- **Buy** when the price closes above the highest close of the previous `breakout_days` (20), while the 50-day average is above the 200-day average.
- **Sell** when the price closes below the lowest close of the previous `trail_days` (40). This trailing exit lets winners run. Also sell on the `stop_loss` (15%) or after `breakout_max_hold_days` (40).

**pullback**
- **Buy** when the price is above its 200-day average and RSI(3) is below `entry_rsi` (a sharp dip in an uptrend).
- **Sell** on a close above the 10-day average, the stop, or after `max_hold_days`.

Common rules:
- Up to `max_positions` (3) at once, each using one third of the sleeve. If more signals fire than there are open slots, pullbacks are bought first, then the strongest breakouts.
- `use_stocks = true` adds `stock_universe`, the largest US and Canadian companies **as of 2012**. This avoids flattering the backtest with today's winners.
- **No new entries** while `market_filter_ticker` (SPY) is below its 200-day average.
- `execution_lag_days = 1` means the signal comes from yesterday's close and the trade happens at today's close. That's the strict, no-look-ahead version.

Why breakouts only? The optimizer found that breakouts made money in both 2012–2020 and 2021+ (about +2% per trade, 57% winners). Pullbacks barely beat trading costs. Breakout trades last about 5–7 weeks on average, and shorter exits (a 10-day trailing low) did worse because they cut the big winners short.

## Daily Scanner (`scan.py`)

**Discovery**: by default the scanner pulls the current S&P 500 and TSX 60 member lists (`scanner.discover_indexes`) and checks all ~580 stocks. That way it finds setups in companies you don't already follow. They're tagged **NEW** and shown with company name and sector. The member lists come from Wikipedia and are cached in `data/universe_*.csv` for a week. If Wikipedia can't be reached, the last saved copy is used.

**Industry breakdown**: every scan starts with a table showing, for each industry, how many stocks were scanned and how many have a buy signal. It also shows the **% signalling**, the average 3-month momentum, and the top pick. A high % signalling means the whole industry is moving, not just one stock. The buy shortlist is capped at `max_per_sector` (3) per industry, so it stays diversified even when one industry dominates. TSX sector names are mapped to the S&P's names so the industries line up. The table is also saved to `outputs/scan_<date>_industries.csv`.

**Industry explorer (dashboard)**: a table of every industry showing buy signals today, % of stocks in an uptrend, median 3-month move, your dividend weight and income, and swing history. **Click any industry** to open its breakdown:
- every stock the scanner checked in it, sortable by signal, 3-month move, or distance above the 200-day average, with a "YOU OWN" tag on your holdings
- your dividend holdings there
- how swing trades have done there in the backtest (win rate, average trade, best and worst stocks)

Clicking an industry in *Industries today* or *Dividend portfolio* opens the same view. You can bookmark one with `#industry=Energy` at the end of the address. In the terminal, `python scan.py --industry energy` prints the same stock-by-stock breakdown. The filter is a partial match, so `health` works.

The scanner uses **completed closes only**. If you scan during market hours, it uses yesterday's close, the same as the trader, so the two always agree.

`python main.py` also ends with a **dividend core by industry** table: each industry's weight, yield, annual income, share of total income, and holdings.

The scanner downloads fresh data in each stock's own currency and lists:

- **BUY SIGNALS**: stocks that triggered a setup at the latest close. These are ranked by signal strength plus how well the setup has worked on that stock before (a record with more trades counts for more). The top `max_buy_ideas` are shown.
- **WATCHLIST**: tickers close to triggering (within 2% of a breakout, or RSI near the entry level).

For each one it shows the stop price, the exit rule and level, a share count sized from `scanner.account_size`, and liquidity in $ per day. It also shows:

- **Track record**: how this exact setup has done on this ticker over the last `history_years`, before costs. A ticker with a poor record (e.g. 13% winners) is worth skipping even when it signals.
- **Earnings**: the next earnings date, flagged `EARNINGS SOON` within `earnings_buffer_days`. Earnings gaps can blow straight through a stop.

To check exits on trades you've taken, create `my_swing_positions.csv`:

```text
ticker,entry_date,entry_price,setup
XLF,2026-11-03,57.50,breakout
```

Each run prints HOLD or SELL (with the reason) for every row. Results are saved to `outputs/scan_<date>.csv`.

## Dividend Core Rules

At each `core.rebalance_frequency` date, every stock is scored **within its own sector**:

| Factor | Weight | What it rewards |
|---|---|---|
| Dividend yield | 30% | Higher indicated yield (yields above `max_yield` are treated as traps) |
| Dividend growth | 25% | Average payment this year vs last year |
| Low volatility | 15% | Steadier stocks |
| Trend | 30% | 12-month total return and being above the 200-day average |

The top `stocks_per_sector` names in each sector are held. To keep turnover low:

- A current holding is kept while it ranks within the top `stocks_per_sector + hold_buffer`.
- A holding is only traded when its weight drifts more than `rebalance_band` from target.
- **Dividend cuts**: a stock whose dividend falls more than `cut_threshold` is sold and not bought again until it recovers.

Dividends are handled by `dividends.mode`:

- `drip`: reinvested into the stock that paid them, commission-free like a broker DRIP (default).
- `smart`: pooled and invested into the most underweight holdings.
- `cash`: withdrawn as income.

`dividends.us_withholding_tax = 0.15` applies the 15% U.S. withholding tax to U.S. dividends (TFSA or taxable). Set it to `0.0` to model an RRSP.

## Where To Edit Settings

Everything is in [config/settings.toml](config/settings.toml) and commented inline. The common ones:

- `swing.capital_fraction`: how much starts in the swing sleeve
- `profit_sweep.fraction`: share of net swing profit moved to the dividend core
- `[sectors]` and `[target_allocation]`: the dividend universe and industry weights
- `transaction_fee`, `slippage_bps`, and `swing.slippage_bps`: trading costs

## Optimizer

`optimize.py` randomly samples the `SEARCH_SPACES` at the top of the file. It **picks settings using only data before `out_of_sample_start`** (default 2021), then reports how those settings did afterwards.

- `--focus swing` scores the swing engine on its own results: the sleeve's value plus everything it swept into the dividend core.
- `--focus core` scores the whole portfolio, and also runs a "no swing" baseline.

It prints the in-sample vs out-of-sample rank correlation. Near zero means the top runs won mostly by luck. The swing search scored 0.59, which suggests the swing tuning found something real. Use the **robust choices**, the parameter values that averaged best in both periods, rather than the single top run. `outputs/best_optimization_summary_<focus>.txt` has TOML snippets for both that you can paste into settings.

## Output Files

- `equity_curve.csv`: daily total, core, and swing value, cash, dividends, and profit swept
- `trades.csv`: every simulated trade with sleeve and reason (`entry_signal`, `stop_loss`, `deploy`, `dividend_cut`, ...)
- `swing_trades.csv`: each swing round trip with P&L, holding days, and exit reason
- `positions.csv`: final holdings with dividend yield and forward annual income
- `dividend_income_by_year.csv`: dividends received per year
- `performance_summary.csv`: CAGR, drawdown, Sharpe, benchmark, swing stats, forward dividend income
- `sector_weights.csv`: daily industry weights plus swing sleeve and cash
- `optimization_results_<focus>.csv` and `best_optimization_summary_<focus>.txt`: from `optimize.py`
- `scan_<date>.csv`: from `scan.py`

## Realistic Execution (backtest = live)

The backtest follows the same rules a real account and `live_trader.py` must follow:
- Orders are sized from the **previous close** and filled at today's close (market-on-close).
- Buys only use cash on hand **before** the day's sales. Sale proceeds aren't known until the close and settle the next day.
- **One fee per order**, even if a stock is topped up twice in one day.
- A stock sold today isn't bought back the same day.
- Swing profits are swept **once per day**, net of that day's losses.
- Ties go alphabetically.

`audit_backtest.py` replays the live trader one day at a time over 2023–2024 for both the C$100k research account and your plan. Both match the backtest trade for trade.

## Known Limitations

- **Hindsight in the stock list.** The dividend universe was chosen in 2026 from companies that are well-known dividend payers today, so the backtest flatters the core. Treat results as an upper bound. The swing stock list uses 2012's largest companies to limit this, but companies that disappeared since (e.g. Potash Corp) are missing.
- The scanner's per-ticker track record covers only a few years and a handful of trades per ticker. Treat it as a sanity check, not a forecast.
- Dividends are credited on the ex-dividend date instead of the pay date.
- Canadian and U.S. holidays differ, so a stock can occasionally "trade" at the previous day's price.
- No capital gains tax is modeled, which matters for the swing sleeve in a taxable account.
- `cash_interest_rate` is a constant; real rates ranged from about 0.5% to 5% over the period.
