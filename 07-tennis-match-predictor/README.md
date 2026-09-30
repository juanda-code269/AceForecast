# RallyOdds

Chronologically validated tennis match probabilities using logistic regression and random forest, with calibration, player surface summaries, matchup simulation, and feature importance.

The bundled tour is synthetic. Upload dated match-level data with two players, binary outcome, and pre-match numeric features.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Do not compute features using post-match information; that creates leakage.

