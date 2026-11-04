# Your AI Investor

A machine-learning investor trained **on your computer** from 16+ years of market data. No API keys, no outside AI services, nothing uploaded.

## Commands (run from the project folder)

```bash
python -m ai_investor.train                  # train / retrain the AI (first run downloads ~580 stocks, a few minutes)
python -m ai_investor.advisor                # today's research report and plan
python -m ai_investor.advisor --no-news      # same, skipping the news check
python -m ai_investor.news --tickers CNQ.TO  # read and score headlines for any stocks
python -m ai_investor.news --train           # train the news model on headlines you've collected
python -m ai_investor.forward                # the AI's fake-money account: run once per trading day
python -m ai_investor.forward --status       # what the AI's account holds
```

## AI forward test (`forward.py`)

The AI runs a second **fake-money account** next to the rules forward test. Everything except the stock picking is the same:
- the money: C$1,000 start plus monthly deposits, the 25/75 split, fees and whole shares
- the profit sweep, the 15% stop-loss, and the pause in market downtrends

So the dashboard comparison shows which **picks** stocks better.
- **Swing:** buys the AI's top-ranked large companies and holds each for about a month (its prediction horizon).
- **Dividends:** the AI's best dividend payer per industry (half AI one-year view, half dividend quality), as many as the account can hold, re-checked quarterly. Dividends are reinvested.

`run_daily.bat` runs both accounts. Files are in `live/` with `_ai` in the name. Judge it over months, not weeks.

Retrain about once a month (`train`), and run the advisor daily or whenever you're about to trade.

## What's inside

| File | What it does |
|---|---|
| `dataset.py` | Downloads prices and dividends (S&P 500 + TSX 60 + your lists, in CAD) and computes ~30 inputs per stock per week: momentum, trend, breakouts, RSI, volatility, beta, dividend yield and growth, market conditions, industry |
| `train.py` | Trains the **swing model** (next ~month) and the **long-term model** (next ~year), tests them walk-forward, saves them to `models/`, writes `reports/training_report.md` |
| `news.py` | Free Yahoo Finance headlines, scored locally with a finance word list. Flags red-flag phrases (dividend cut, downgrade, lawsuit...). Saves every headline to `data/news_archive.csv` |
| `advisor.py` | The daily report: market check, swing plan, long-term dividend plan by industry, news check, and how much to trust it. Writes `reports/plan_<date>.md` |
| `config.toml` | Horizons, universe, how many ideas to show |

## How the AI works

Both models are **gradient-boosted decision trees** (scikit-learn). Each learns to predict how a stock will rank against all other stocks over its horizon, from the inputs above. It predicts a **ranking, not a price**, which is both more useful for picking stocks and more stable to learn.

They're deliberately conservative (few, large decision leaves; strong regularisation). Stock returns are mostly noise, and a flexible model memorises noise.

**How the plan uses them:**
- **Swing:** your *tested breakout rules* find today's setups, and the AI ranks them. The AI never invents swing trades the rules didn't signal (those go on an "AI watchlist" for information).
- **Long-term:** dividend payers that pass your yield and dividend-cut rules are ranked by the one-year model **within each industry**, so the plan stays diversified. Your account size decides how many to hold.
- **News:** the top ideas and your holdings get a headline check. Negative news or red flags put a ⚠ on the idea; nothing is bought or sold on news alone.

## How it was tested (and why some numbers in the report look too good)

**Walk-forward:** for each year from 2016, a model trained *only on earlier years* predicted that year. Training labels end more than a full horizon before the test year starts, so no future information leaks in.

The report shows three tests:
1. **Ranking skill** vs a simple momentum rule (buy last year's winners). This is the cleanest measure.
2. **An investing test on today's S&P 500 + TSX 60.** Its absolute returns are inflated by **survivorship bias**: the list only contains companies that succeeded, so "buy the hottest stocks" looks amazing in hindsight. Use it only to compare strategies with each other.
3. **The same test on companies that were already large in 2012**, plus **the check that matters most**: did the AI's ranking improve the breakout trades your rules actually made?

## Optional: a free news-data key (Finnhub)

Data APIs are fine (only outside *AI* services are off-limits). With a free key from finnhub.io, the news reader uses Finnhub's company news first. It can also download about a year of past headlines, so your news model can train **now** instead of after 1–2 months of collecting:

```powershell
setx FINNHUB_API_KEY your_key_here      # once; then open a new terminal
python -m ai_investor.news --backfill   # ~1 year of headlines for your stocks, then trains the news model
```

The key is read from the environment and never saved in any file. Without a key, news comes from Yahoo Finance, Yahoo's RSS feed, or Google News, whichever is working.

## How it learns over time

- **Prices:** `train` re-learns from the newest data each time you run it.
- **News:** every headline the advisor reads is saved. Free sources only give recent headlines, so there's no news history to learn from yet. After roughly 1–2 months of daily use, `news --train` can learn which words actually moved prices. The trained model is only used if it beats the word list on the newest headlines.

## Limits

- The AI **advises**; it doesn't trade. `live_trader.py` keeps trading the tested rules.
- A model can't see the future. Most of any single stock's move is luck.
- The news score reads headlines only, and can't be checked against history until your archive grows.
- Not financial advice.
