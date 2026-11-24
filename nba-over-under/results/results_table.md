Test season games: 1264 (1222 with a valid sportsbook line). Break-even win rate at -110 pricing: 52.38%.

| Model | Val MSE | Test MSE | Test MSE (games with line) | Mean error (bias) | Beats line | 95% CI | Under bets | ROI @ -110 | Paper test MSE |
|---|---|---|---|---|---|---|---|---|---|
| Predict training mean | 506.2 | 519.5 | 518.3 | -11.7 | 53.2% (1205 bets) | 50.4–56.0% | 92% | +1.6% | - |
| Season-average heuristic | 332.5 | 347.9 | 349.5 | -1.0 | 50.3% (1203 bets) | 47.5–53.1% | 56% | -4.0% | - |
| Random Forest | 374.3 | 380.9 | 380.4 | -4.7 | 52.9% (1205 bets) | 50.1–55.8% | 72% | +1.1% | 910.3 |
| Collaborative Filtering (biased MF) | 328.6 | 359.5 | 362.8 | -0.6 | 48.0% (1205 bets) | 45.2–50.9% | 54% | -8.3% | 459.9 |
| Neural Net (MLP) | 342.5 ± 1.8 | 354.8 ± 2.0 | 354.9 ± 2.2 | +0.7 | 53.0% (1205 bets) | 50.2–55.8% | 44% | +1.2% | 369.8 |
| LSTM | 354.2 ± 4.2 | 368.2 ± 3.9 | 367.4 ± 3.6 | -2.9 | 53.0% (1205 bets) | 50.2–55.8% | 65% | +1.2% | 426.6 |
| Sportsbook line | - | 332.1 | 332.1 | -0.0 | - | - | - | - | 320.7 |
| Always bet Under | - | - | - | - | 51.6% (1205 bets) | 48.8–54.4% | 100% | -1.5% | - |
| Always bet Over | - | - | - | - | 48.4% (1205 bets) | 45.6–51.2% | 0% | -7.6% | - |
