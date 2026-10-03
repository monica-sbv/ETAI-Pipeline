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


## Week 3 - EDA_cleaning

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

## Week 4 - Preprocessing and Cross Validation

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
