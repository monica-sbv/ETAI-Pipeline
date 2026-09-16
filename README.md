# Baseline Predictive Pipeline -- ETAI

## Project structure

```
.
├── main.py                # entry point: run the whole pipeline
├── config.yaml             # all tunable settings live here
├── requirements.txt
├── src/
│   ├── data.py             # loading
│   ├── preprocessing.py    # cleaning + train/test split
│   ├── model.py             # model construction
│   ├── evaluate.py         # accuracy metrics + fairness check
│   └── results.py          # saves each run's report to disk
├── results/                # created automatically -- one file per run (not tracked in git)
└── data/
    ├── compas_two_year_recidivism.csv
    └── README.md            # problem description + full data dictionary
```

## Pipeline progress

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |


# 1st commit task

Number: 20260542
Name: Mónica Vilela

# Accuracy

Logistic regression:
Train accuracy: 0.679
Test accuracy:  0.677
Gap (train - test): +0.002

Decision tree:
Train accuracy: 0.680
Test accuracy:  0.668
Gap (train - test): +0.012

This means that the Logistic Regression has a slightly higher test accuracy and the gap between the test and train accuracy is only 0.002 compared to the 0.012 of the decision tree, which means there is less (almost no) overfitting.

0.67 of teste accuracy means the model got 67 out of 100 test examples correct.