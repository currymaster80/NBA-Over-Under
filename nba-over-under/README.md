# Predicting NBA Over/Under totals with machine learning

A from-scratch, tested re-implementation of the models in **"The Bank is Open: AI in Sports
Gambling"** (Bucquet & Sarukkai, Stanford CS229). For every NBA game we predict the **total
points scored by both teams** and ask two questions:

1. How close are we to the number a sportsbook posts (the *Over/Under line*)?
2. If we bet Over/Under against that line, would we win often enough to profit?

**In plain English.** Before each game, a bookmaker publishes a guess such as "the two teams
will score 215 points combined". You can bet that the real total will be higher (*Over*) or lower
(*Under*). Our models make their own guess from each team's last three games, how strong their
opponents were, how rested they are and how far they travelled. If our guess is better than the
bookmaker's, we win more bets than we lose. Spoiler: the bookmaker's guess is hard to beat.

## Results (test season 2017-18, 1,264 games)

Trained on 2012-13 → 2015-16, tuned on 2016-17, evaluated once on 2017-18. Lower MSE is better
(MSE in points²; "beats line" = share of bets won against the sportsbook line, pushes excluded).

| Model | Test MSE | Beats line (95% CI) | Under bets | ROI @ −110 | Paper's test MSE |
|---|---|---|---|---|---|
| Sportsbook line | **332.1** | – | – | – | 320.7 |
| Season-average heuristic (no learning) | 347.9 | 50.3% (47.5–53.1) | 56% | −4.0% | – |
| Neural net (MLP 500-100-20-1), 5 seeds | 354.8 ± 2.0 | 53.0% (50.2–55.8) | 44% | +1.2% | 369.8 |
| Collaborative filtering (biased MF) | 359.5 | 48.0% (45.2–50.9) | 54% | −8.3% | 459.9 |
| LSTM, 5 seeds | 368.2 ± 3.9 | 53.0% (50.2–55.8) | 65% | +1.2% | 426.6 |
| Random Forest | 380.9 | 52.9% (50.1–55.8) | 72% | +1.1% | 910.3 |
| *Predict training mean (zero skill)* | 519.5 | 53.2% (50.4–56.0) | 92% | +1.6% | – |
| *Always bet Under* | – | 51.6% (48.8–54.4) | 100% | −1.5% | – |

The full table (validation MSE, bias, per-model details) is in `results/results_table.md`; figures
are in `results/fig_*.png`. Break-even at standard −110 pricing is a **52.38%** win rate.

**What to take from this**

- **No model beats the sportsbook.** The best learned model (MLP) is 22.7 points² worse than the
  line (354.8 vs 332.1), the same ordering the paper found, with a smaller gap.
- **A one-line heuristic beats every learned model.** Averaging each team's season-to-date game
  totals scores 347.9. The networks hit their best validation error after about 3 epochs and
  then overfit the ~5,000 noisy training games (`results/fig_training_curves.png`).
- **The "win rate vs. the line" cannot be trusted at this sample size.** About 1,200 bets give a
  ±2.8-point confidence interval, and a model with **no skill** (always predict the training mean)
  "wins" 53.2% because totals rose from ~201 (training) to ~212 (test), so it almost always bets
  Under. None of the intervals separates a real edge from luck or from that Over/Under lean, and
  none of them clears the 52.38% break-even with confidence. **This project does not show a
  profitable betting strategy.**
- **The NBA changed under the models' feet.** Mean game totals rose from 195.9 (2012-13) to 212.4
  (2017-18) as teams shot more threes. Models trained on older seasons carry that bias (see the
  "Mean error" column in `results/results_table.md`).
- **For collaborative filtering, latent factors did not help.** Validation preferred
  `reg_factor=1000` (factors switched off, i.e. rolling team home/away scoring offsets) over any
  setting with factors on (best 401.6 vs 328.6 validation MSE).

## Quick start

Tested with Python 3.13, NumPy 2.5, pandas 3.0, scikit-learn 1.9, PyTorch 2.14 on CPU.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

pytest -q                                              # 13 unit tests (~2 s)
python scripts/run_pipeline.py --quick                 # demo run, about 30 s on a laptop CPU
python scripts/run_pipeline.py                         # full run (grids + 5 seeds), about 13 min on 2 CPU cores
```

Outputs land in `results/`: `results_table.md`, `metrics.csv`, `test_predictions.csv`,
the hyper-parameter grids (`*_grid.csv`), `run_info.json` (chosen hyper-parameters, split sizes,
runtime) and three figures. Useful flags: `--models rf,cf,mlp,lstm` (subset), `--seeds N`,
`--out-dir DIR`, `--history K` (games of history per team). See `python scripts/run_pipeline.py -h`.

`--quick` uses single small grids and 2 seeds, so its numbers differ slightly from the full run.
The tables above come from the full run. Runs are seeded; the neural models are reproducible on
the same machine and library versions, but may differ slightly across hardware or PyTorch versions.

## Repository layout

```
├── README.md
├── requirements.txt
├── data/
│   ├── final_game_data/       # box scores, one row per team per game (2007-08 … 2017-18, no 2011-12)
│   └── odds_data_processed/   # sportsbook lines + final totals, one row per game
├── src/nbaou/
│   ├── data.py                # loading, cleaning, team-name harmonisation, odds join
│   ├── features.py            # leakage-free features, train/val/test split, scaler
│   ├── baselines.py           # mean predictor, Random Forest (+ grid search)
│   ├── collab.py              # rolling biased matrix factorisation (collaborative filtering)
│   ├── nets.py                # MLP, LSTM, shared trainer with early stopping, grids
│   ├── evaluate.py            # MSE, betting simulation, Wilson confidence intervals
│   ├── report.py              # results table + figures
│   └── pipeline.py            # end-to-end experiment
├── scripts/
│   ├── run_pipeline.py        # command-line entry point
│   └── download_data.py       # optional: re-fetch the raw data from the original repository
├── tests/test_pipeline.py     # leakage, split sizes, join consistency, betting maths, …
└── results/                   # outputs of the full run reported above
```

## Data

| Source | What | Used for |
|---|---|---|
| Basketball Reference (via the original project's scraper) | Box-score statistics for every team in every game | Features and the target (total points) |
| Sports Book Review Online (via the original project) | Opening Over/Under line for each game | **Evaluation only**, never a model input |

The CSVs are bundled so the project runs offline (≈24 MB). They are the processed files from the
original authors' public repository, <https://github.com/abucquet/bank_is_open>, which does not
state a licence; keep this repository private as the course requires, and cite the sources.
`python scripts/download_data.py` re-fetches them (verified to reproduce the bundled files exactly).

Cleaning decisions (all in `data.py`):

- Dates are stored as `YYYYMMDD` plus one extra digit; the extra digit is dropped.
- Basketball Reference and the odds site use different team names (e.g. "Los Angeles Lakers" vs
  "LA Lakers"); both are mapped to one code per franchise.
- 5 odds rows have impossible full-game lines (19.5, 20.5, 43.5, 47, 51 – evidently data-entry or
  scraping errors); lines outside 150–280 are dropped.
- A line is kept only if the odds file's final total equals the box-score total (13 games
  disagreed and lose their line). Overall 97.5% of games keep a valid line; the rest are still
  predicted and scored by MSE, just not by the betting metrics.

## Method

**Features** (`features.py`). For a game between home team H and away team A, each team
contributes its previous 3 games of the same season (oldest → newest). Every past game provides the
92 box-score statistics of that game plus the **opponent's season-to-date averages** of those
statistics (computed only from games before that past game, so strength of schedule is captured
without peeking ahead). Static features: days of rest and miles travelled for both teams, and
one-hot team identities. Total input size: 1,170 features. The first three games of a team's
season are dropped because they lack three games of history, which reproduces the paper's split
sizes exactly (5,069 / 1,260 / 1,264). A unit test confirms that deleting all future games
leaves every feature unchanged.

**Models** (`baselines.py`, `collab.py`, `nets.py`)

| Model | Idea | Tuned on validation |
|---|---|---|
| Random Forest | Baseline on the flattened features | `max_features`, `min_samples_leaf` (300 trees) |
| Collaborative filtering | League as a "recommender": rows = home teams, columns = away teams, rating = total points. For each game date, fit `μ + bᵢ + cⱼ + uᵢ·vⱼ` on games in the preceding window, then predict the matchup | Window (1–90 days), regularisation, factors on/off |
| Neural net | Fully connected 1170→500→100→20→1, ReLU, as in the paper | learning rate, weight decay, dropout |
| LSTM | One LSTM layer reads the 3 past games one at a time; final hidden state + static features → linear output | learning rate, hidden size (20/40), dropout |

Neural models regress the standardised total with MSE, AdamW and early stopping on validation MSE,
and are trained with 5 seeds; mean ± std over seeds is reported. All scaling statistics come from
the training seasons only.

**Evaluation** (`evaluate.py`). MSE against the true total, and the paper's "beat the house"
simulation: bet Over when the prediction is above the line and Under when it is below. Pushes
(total equals the line) are refunded and excluded; a game where the prediction equals the line
is not bet. ROI assumes every bet is priced at −110 (win +0.909, lose −1). Intervals are 95%
Wilson intervals.

## Differences from the paper (and why)

- **Optimiser and scaling.** The paper trained the network with plain gradient descent (learning
  rate 1e-6, weight decay 1, 5,000 epochs) on unscaled inputs. Here inputs and targets are
  standardised and training uses AdamW with early stopping, which is far more stable.
- **Collaborative filtering** is implemented directly in NumPy (biased matrix factorisation, the
  same model family as the Surprise SVD the paper used) rather than via the Surprise package,
  which is awkward to install on recent NumPy versions.
- **Feature set.** The paper's exact feature list is not documented (its input has 1,524
  dimensions); we use all 92 box-score statistics the data provides, giving 1,170 inputs. The
  opponent season averages are computed point-in-time (no leakage).
- **LSTM head** also receives the static features (rest, travel, team identities).
- **Hyper-parameter search** uses small validation grids for every model, so all models get a
  comparable amount of tuning. The paper's Random Forest (MSE 910) looks untuned; ours is 380.9,
  so our gaps between models are smaller than the paper's.
- **Sportsbook MSE.** The paper reports 320.7; on our games-with-a-valid-line subset the line
  scores 332.1. We do not know exactly which games the paper scored, so compare ordering and
  rough magnitude rather than third-digit equality.
- **Extra diagnostics** not in the paper: reference predictors, always-Under/Over rows, an
  Under-share column, confidence intervals and ROI, because raw win rates are easy to
  over-read (see Results).

## Limitations

- One test season (~1,200 bets) is a small sample for betting conclusions; no model was run on
  other seasons, and the validation season was also used to choose between models.
- Distance uses straight-line miles between arena cities (approximate coordinates), and the
  rest/travel features only look at the previous game, not cumulative road-trip fatigue.
- Injuries, lineups and closing-line movement are not modelled; these are where the paper's own
  discussion suggests improvement.
- The collaborative-filtering validation error was still edging down at the 90-day limit of the
  grid. A one-off probe of longer windows up to the full season-to-date gave 327.2–328.0
  validation MSE, so the optimum is a plateau, not a cliff. To explore it yourself, extend
  `windows` in `collab.tune_and_predict`.

## Credits

Original project: Alexandre Bucquet and Vishnu Sarukkai, *The Bank is Open: AI in Sports
Gambling*, Stanford CS229 (2018). Data: Basketball Reference and Sports Book Review Online, as
collected in the original repository. This repository re-implements the methods independently
for a university machine-learning mini-project (UE24CS352A).
