# Pairs Trading Backtest

This is a project where I tested a pairs trading idea on old stock data to see if it would have made money. I also tried adding a logistic regression to skip bad trades.

This project was made using help from AI.

## What is pairs trading?

Some stocks usually move together. Sometimes one gets too expensive compared to the other. The bet is that the gap will close again. So when the gap gets big, I buy the cheaper stock and short the expensive one, then close the trade when the gap shrinks.

## How it works

I used 30 big US financial stocks and split the data into three time periods that don't overlap:

1. 2016-2018: pick the 10 pairs of stocks whose daily moves are most correlated. I use a line of best fit (linear regression) to get the hedge ratio, which says how much of one stock to trade against the other. After this period the ratio stays fixed.
2. 2019-2020: train the logistic regression.
3. 2021-2025: test everything here. Nothing was tuned on this data.

The trading rules:
- The spread is the gap between one stock and where the best fit line says it should be.
- The z-score is how far the spread is from its 60-day moving average, measured in standard deviations.
- Open a trade when the z-score goes past 2 or below -2.
- Close it when the z-score gets back within 0.5 of zero.
- Close it early if the z-score goes past 4 (it's getting worse) or after 40 days.

The logistic regression looks at a stretched spread and guesses if the trade will make money in the next 5 days. If the guess is too low, the trade is skipped.

## How to run it

    pip install -r requirements.txt
    python run.py
    or 'py run.py' if the above doesn't work

## Results (test period 2021-2025, after trading costs)

| Strategy | Return per year | Sharpe | Worst drop |
|---|---|---|---|
| Pairs trading | 0.5% | 0.19 | -6.2% |
| Pairs trading + logistic regression | 0.3% | 0.17 | -3.4% |
| SPY (buy and hold) | 13.6% | 0.80 | -26.2% |

The strategy did not work. It made 2.4% total over 5 years, while SPY made 83%. The logistic regression did not help either. Its AUC was 0.475, which is no better than a coin flip, and it was less accurate than just saying yes to every trade. It had a smaller worst drop, but mostly because it made fewer trades.

I think a big reason is that correlation only checks if two stocks move together day to day. It doesn't check that the gap between them stays stable, so some pairs drifted apart and lost money.

## Things I did to keep the test fair

- A position only earns the next day's return, so it can't use future prices. There are tests for this.
- I counted trading costs (0.05% each time a position changes).
- I removed the last 5 days of the training data, because their answers depend on prices from the test period.
- I picked the pairs before training the model.
- I compared the model against always saying yes.

## Limits with this project

- Only one test period and only 10 pairs.
- The stock list is today's list, so it leaves out companies that failed or got bought. This makes results look better than they should.
- No fees for shorting stocks and no price impact from trading.
- Pairs can stop moving together, and the stop loss only helps a little.

## Files

    run.py           runs everything
    src/data.py      downloads prices, makes fake test data
    src/pairs.py     picks pairs, builds spreads and z-scores
    src/backtest.py  trading rules, returns, results
    src/ml.py        logistic regression
    tests/           checks that the code doesn't use future data