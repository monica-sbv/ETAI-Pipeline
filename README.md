# Baseline Predictive Pipeline -- ETAI

### Project structure

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

### Pipeline progress

| Week | Practical class focus | Added to the pipeline |
|------|------------------------|------------------------|
| 2 | Introduction & baseline pipeline | Initial version: project structure, a single naive train/test split (no cross-validation), minimal preprocessing (drop rows with missing values, one-hot encode categoricals), logistic regression baseline, a first (deliberately simple) fairness check comparing our model's and COMPAS's own false-positive rate by race, train-vs-test accuracy reporting (to start spotting overfitting), and each run's full report saved automatically to `results/` |


## 1st commit task

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


### Week 3 - EDA_cleaning

# Accuracy comparison

| Modelo | Estado do Data | Train Accuracy | Test Accuracy |
| :--- | :--- | :--- | :--- |
| **Logistic Regression** | **Antes (Raw)** | 0.679 (67.9%) | **0.677 (67.7%)** |
| **Logistic Regression** | **Depois (Cleaned)** | 0.678 (67.8%) | **0.655 (65.5%)** |
| **Decision Tree** | **Antes (Raw)** | 0.680 (68.0%) | **0.668 (66.8%)** |
| **Decision Tree** | **Depois (Cleaned)** | 0.691 (69.1%) | **0.642 (64.2%)** |

Impacto das alterações com cleaning:

-Queda da Test Accuracy poderá ter acontecido devido à eliminação dos duplicados e redundâncias. O dataset raw tinha 72 linhas duplicadas e colunas redundantes com multicolinearidade perfeita e quando isso acontece os conjuntos de treino e teste partilham informação quase idêntica o que vai inflacionar a test accuracy de forma irrealista. Esta queda não significa totalmente que o modelo piorou depois de fazer a cleaning, significa que agora a métrica apresenta resultados mais honestos do desempenho do modelo com dados reais. (como ainda não fiz o preprocessing, por agora ainda está a baseline naive de "dropna()" em vez de fazer imputação, o que pode estar a eliminar linhas com informação útil)

Comparação de modelos:

-O melhor modelo continua a ser o Logistic Regression com 65.5% de test accuracy e decisio tree com 64.2%. O gap entre treino e teste na Logistic Regression é +0.023, enquanto na Decision Tree é de +0.049. Comparando com os valores da semana passada, estes valores não demonstram que os modelos passaram de ter um problema de overfitting mas que os conjuntos de treino e teste são agora um pouco mais independentes. Apesar de os valores terem subido, os valores atuais são bons e indicam modelo estável (abaixo de 5%). Por isso, na semana passada havia uma falsa sensação de estabilidade e agora foi possível eliminar algum vício dos dados.

### Week 4 - Preprocessing and Cross Validation

# LR last week VS. LR this week

**Last Week (Single split on cleaned data):**

Train accuracy: 67.8%
Test accuracy: 65.5%
Train-test gap: +2.3%

**This Week (5-Fold CV):**

Mean train accuracy: 67.5% (std = 0.003)
Mean validation Accuracy: 67.2% (std = 0.013)
Mean overfitting gap: +0.3% (std = 0.016)

*Conclusion:*

The performance on training data remains virtually identical, but the single test split last week underestimated performance on test data, with CV giving a more reliable 67.2% mean. The small validation std and tight gap show that LR is very stable and leak-free across different folds and is overfitting is minimal.

Also, precision, recall and FPR also got slightly better which shows ...

# Models performance comparison this week

| Modelo | Mean train acc | Mean val acc | Overfit gap | F1 (0/1) | Std Val|
| :--- | :--- | :--- | :--- | :--- | :--- |
| Dummy | 54.9% | 54.9%  | 0.0%  | 0.71/0.00 | 0% |
| LR | 67.5% | 67.2% | 0.3% | 0.73/0.58 | 1.3% |
| DT | 68.5% | 67.5% | 1.1% | 0.73/0.60 | 1.8% |
| RF | 73.3% | 65.0% | 8.3% | 0.69/0.60 | 1.8% |

*Takeaway:*
LR continues begin the better model, because:
- It has the best generalization, meaning the smallest train-validation gap;
- It has the lowest standard deviation across validation folds making its predictions the most reliable;
- It outperforms RF on validation accuracy because RF overfits the training set (gap of 8.3%);
- Matches DT but with less variance and tigher generalization gap.

### Week 5 - Hyperparameter Tuning

| Model | Default: CV mean ± std | Tuned: nested CV mean ± std | Tuning score (best trial) | Optimism | Chosen hyperparameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| DT | 0.675 ± 0.018 | 0.672 ± 0.017 | 0.679 | +0.002 | max_depth: 6, min_samples_leaf: 101, criterion: gini |
| LR | 0.672 ± 0.013 | N/A (tuning off) | N/A | N/A | max_iter: 2000 (default) |
| RF | 0.650 ± 0.018 | 0.681 ± 0.015 | 0.683 | +0.000 | max_depth: 29, min_samples_leaf: 63, max_features: sqrt |

# Analysis

- Model gains and standard deviation: Random forest gained from tuning, jumping from a default validation score of 0.650 up to 0.681, a gain of +0.031, which exceeds its fold-to-fold validation standard deviation (std= 0.015). In contrast, decision tree showed no significant gain, scoring 0.675 by default and 0.672 after nested CV tuning.
- Impact on model ranking: Before tuning, default random forest lagged behind both logistic regression (0.672) and decision tree (0.675) due to unoptimized defaults. After tuning, random forest edged past them to become the top-performing model with an honest nested CV score of 0.681.
- Why: A single decision tree is prone to high variance and easily overfits training noise. By averaging the predictions of 300 individual trees, random forest dramatically reduces prediction variance while maintaining low bias, creating a much more robust decision boundary. Tuning via Optuna identified optimal parameters (max_depth: 29, min_samples_leaf: 63, max_features: 'sqrt') that prevented individual trees from memorizing noise.
- Reporting metrics: For the best model (Random Forest), we should report the honest nested CV validation score of 0.681 ± 0.015 rather than the optimistic best-trial tuning score (0.683). Nested cross-validation is very important because it prevents selection bias and it ensures that the performance estimate is unbiased and generalizable to unseen data.   


### Week 6 - Ensembles and Feature Engineering

## Updates based on challenges from previous week

- Added a search space for LR --> The parameter tuned was C, which controls the inverse regularization strength, meaning: a smaller C (ex.: 0.0001) applies stronger penalties to coefficients (stronger regularization), forcing the model to keep weights small and reducing the risk of overfitting; a larger C (ex.: 100) reduces regularization, allowing the model to fit the training data more closely. Unlike DT, LR has a much simpler structure (fixed, straight-line decision boundary) and far fewer hyperparameters. Therefore, tuning LR usually has a much smaller performance gain compared to tuning DT or RF. Linear models are already quite stable and less prone to wild overfitting, so tweaking C mostly just fine-tunes the bias-variance tradeoff slightly.
- Changed/Added more parameters to the search space for RF -->
    - n_estimators allows Optuna to find the ideal spot between performance and training speed. More trees generally make the ensemble more stable and accurate, but increase computation time.
    - null (None) for max_depth allows trees to grow fully until leaves are pure / without a depth limit. Unlike a single DT which overfits if allowed to grow infinitely, a RF benefits from deep, fully-grown trees because the final ensemble averages out individual tree variance.
    - max_features (sqrt, log2, 0.5, 1.0) controls how many features each tree considers when deciding on a split. Testing options like "sqrt" or 0.5 forces the trees to look at random subsets of features, decorrelating the trees from one another, which is the reason behind why RF outperform standard DT.

## Added features:

1. Priors per adult year
*Purpose:* Measures recidivism velocity as a continuous scale. It distinguishes between a 40-year-old who accumulated 4 priors slowly over 22 years vs. a 20-year-old who accumulated 4 priors in just 2 years. Normalizing prior offenses by adult exposure time helps trees make cleaner linear cuts.
*Safety:* uses basic arithmetic and boolean conditions on existing columns, which need no training parameters. --> no data leakage

2. High-risk young adult flag
*Purpose:* Captures a non-linear cliff or threshold effect. Criminology research shows that young adults (<= 25) with any prior record behave drastically differently in recidivism risk than older individuals or young individuals with zero priors. Even though a tree could eventually figure this out by splitting age <= 25 and then splitting priors_count > 0 deeper down the tree, giving it a direct binary flag lets the tree capture this high-risk interaction immediately near the root node without wasting tree depth.
*Safety:* uses basic arithmetic and boolean conditions on existing columns, which need no training parameters. --> no data leakage

3. Binned priors category (fixed domain thresholds)
*Purpose:* priors_count is heavily right-skewed. Most people have 0, 1, or 2 priors, but a small number of individuals have 20, 30, or 40+. Without binning, a decision tree wastes multiple split nodes trying to separate extreme outliers (ex.:, splitting priors == 17 vs priors == 18), which leads to overfitting on noise. This will group high values into a single bucket (ex.:, 6+), stabilizing the tree's splits and telling the model: "Once someone has 6 or more priors, treat them uniformly as maximum historical risk."
*Safety:* uses hardcoded, domain-driven bin edges ([-1, 0, 2, 5, np.inf]). I chose these thresholds based on a quick search on the internet about criminal justice. So there weren't any quantiles calculated from the training set's distribution, menaing nothing is learned from the data here either. Therefore, it's completely safe to run before the split --> no data leakage
Enconding: Ordinal. Since prior categories represent a strict ranking of risk (0 < 1-2 < 3-5 < 6+), we can map them to integers (0, 1, 2, 3). This preserves the natural ordering so tree-based models can treat it as a smooth progression.

## Model comparison (with new features):

| Model | Default: CV mean ± std | Tuned: nested CV mean ± std | Tuning score (best trial) | Optimism | Chosen hyperparameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| DT | 0.671 ± 0.011 | 0.668 ± 0.014 | 0.674 | +0.006 | max_depth: 4, min_samples_leaf: 1, criterion: gini |
| LR | 0.676 ± 0.016 | 0.674 ± 0.014 | 0.678 | +0.002 | C: 47.072 |
| RF | 0.646 ± 0.017 | 0.676 ± 0.009 | 0.678 | +0.003 | n_estimators: 127, max_depth: 20, min_samples_leaf: 23, max_features: sqrt |
| ORF | 0.650 ± 0.019 | 0.675 ± 0.013 | 0.680 | +0.004 | max_depth: 28, min_samples_leaf: 27, n_components: 5 |
| AB | 0.674 ± 0.009 | 0.677 ± 0.013 | 0.680 | +0.001 | estimator__max_depth: 4, n_estimators: 220, learning_rate: 0.025 |

# Without new features:

| Model | Default: CV mean ± std | Tuned: nested CV mean ± std | Tuning score (best trial) | Optimism | Chosen hyperparameters |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ORF | 0.647 ± 0.011 | 0.679 ± 0.013 | 0.683 | +0.003 | tree__max_depth: 6, tree__min_samples_leaf: 3, project__n_components: 5 |
| AB | 0.676 ± 0.012 | 0.682 ± 0.017 | 0.684 | -0.000 | estimator__max_depth: 4, n_estimators: 220, learning_rate: 0.025 |

## Analysis

- Impact of Tuning:
    - Ensembles (RF, ORF, AdaBoost) show significant performance jumps when tuned. For example, default RF with new features scored 0.646, but tuning its depth, feature subsets, and leaf sizes boosted it to 0.676.
    - LR and DT gained basically nothing (or even dropped slightly in validation score) from tuning. As mentioned above, linear models have a fixed boundary and simple parameter space, meaning hyperparameter tuning (C) changes their behavior only marginally.
- With vs. Without New Features:
    - Models without the newly engineered features (like AdaBoost scoring 0.682 and ORF scoring 0.679) performed comparably to or slightly higher than their counterparts with the engineered features.
    - Could be because the raw features (age, priors_count, etc.) already contain the core signal, and tree-based ensembles (especially boosted trees and random forests) are powerful enough to discover non-linear splits and interactions natively without needing pre-calculated velocity or boolean flags explicitly handed to them.
- Generalization and Overfitting:
    - Default untuned ensembles (like RF and ORF) showed big train-validation gaps before regularization (training scores near 0.75–0.77 vs validation scores near 0.64–0.65)
    - Tuning min_samples_leaf and max_depth successfully reined this in, closing the generalization gap and creating robust models with stable cross-validation standard deviations (approx. 0.013).
- Fairness & False Positive Rates:
    - Across all models, the False Positive Rate (FPR) for African-American defendants remains higher (approx. 0.27 - 0.29) compared to Caucasian defendants (approx. 0.14 - 0.16). However, all machine learning models tested here significantly improved upon COMPAS's proprietary score, which originally carried an African-American FPR of 0.45.

The new best model is Adaboost (without / with new features) with a nested CV validation score of 0.682