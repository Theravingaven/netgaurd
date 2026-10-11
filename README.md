# NetGuard IDS

Hybrid network intrusion detector (Isolation Forest + Random Forest, fused by a logistic meta-learner),
served as a Flask API with a small dashboard.

```
train.py  ->  models/*.pkl + reports/metrics.json + reports/EVAL.md
   -> predict_api.py (/predict, /predict_csv, /explain, /metrics)
   -> templates/dashboard.html (/dashboard)
```

## Run
```
pip install -r requirements.txt
python train.py path/to/flows.csv      # needs a label/class/target column
python app.py                          # http://localhost:5000/dashboard
curl -X POST localhost:5000/predict -H "Content-Type: application/json" -d '[{"Flow Duration": 1200, ...}]'
```

## Evaluation
See [`reports/EVAL.md`](reports/EVAL.md), regenerated on every `train.py` run so the numbers can't drift
from the code. One headline metric: held-out attack-class F1 with a 95% bootstrap CI. Split is stratified
60/20/20 before any fitting; scaling, feature selection and models use train only; fusion uses validation only.
`train.py` flags results that look too good (F1 >= 0.99, zero FPR, too few test attacks) instead of reporting them.

## "Why flagged"
`/explain` ranks the 3 features furthest from the NORMAL training profile (deterministic). If `ANTHROPIC_API_KEY`
is set, an LLM only rephrases those facts in 2 sentences (`NETGUARD_LLM_MODEL`, default `claude-haiku-5-5`);
otherwise a template is used. The LLM sees no raw traffic and never decides the verdict.

## Limits
Benchmark CSVs, not live traffic. Live capture (`capture.py`) is local-only and its feature names don't match
the CSV-trained model, so it is not part of the evaluated system.
