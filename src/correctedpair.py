import pandas as pd
from src.evaluate import corrected_paired_test

runs = pd.read_csv("results/runs.csv")
folds = [c for c in runs.columns if c.startswith("fold_")]
print(corrected_paired_test(runs.loc[0, folds], runs.loc[1, folds], test_train_ratio=1 / 4))   # rows 0 and 1; 5-fold: 1 / (5 - 1)