# World Cup Performance Analysis & Classification

Analysis of 616 World Cup team records from 1930 to 2022, using pre-tournament form and group stage results to predict final finishing position and classify which teams reach the quarter-finals.

## Dataset

`WorldCup_Stats.csv` — 616 rows, 24 columns, provided by the University of Nottingham during my first year. Each row is one team in one tournament, covering:

- Pre tournament world ranking
- Form over the previous four years (games, wins, draws, losses, goals)
- Group stage results
- Playoff stage results
- Final classification (1st to 32nd)

## Approach

**Exploration** — correlation heatmap across all features to identify relationships with the target and multicollinearity between predictors.

**Cleaning** — removed four teams with anomalous game counts (80+ against a 75th percentile of 48), imputed missing values with the column median, dropped `Team` and `Year` as non predictive identifiers.

**Feature engineering** — created four per game features normalising totals by matches played, since raw totals aren't comparable across teams with different fixture counts. One hot encoded the continent columns, using Africa as the reference in the continent category.

**Regression** — predicted final classification from pre tournament form and group stage results, comparing linear regression against random forest. Playoff columns were excluded to avoid leaking the outcome, since `Playoff_Matches_Played` alone nearly determines the finish. Coefficients were scaled by feature standard deviation to make their effects comparable, and actual vs predicted scatter plots used to compare model fit.

**Classification** — predicted quarter final qualification as a binary target using random forest and logistic regression, evaluated with accuracy, precision, recall and F1, plus a confusion matrix showing the balance between false positives and false negatives. Feature importances were extracted and the three least important removed to test their contribution.

## Results

| Model | R² | MAE | MSE |
|---|---|---|---|
| Linear Regression | 0.48 | 4.50 | 32.98 |
| Random Forest | 0.63 | 3.81 | 23.65 |

| Classifier | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest | 0.81 | 0.74 | 0.76 | 0.75 |
| Logistic Regression | 0.82 | 0.83 | 0.67 | 0.74 |

Pairing predictions with teams reveals the model predicts toward mid table: every actual winner is predicted between 4.9 and 6.2, showing a case of regression to the mean.

Random forest outperformed linear regression on every regression metric, suggesting non-linear relationships a straight-line fit can't capture. The two classifiers trade off against each other — logistic regression is more precise, random forest catches more actual qualifiers.

Group stage goal difference dominated feature importance (0.23), followed by goals scored and conceded. Continent features contributed almost nothing, all below 0.014.

## Limitations

- The dataset combines men's and women's tournaments without a distinguishing column, so the models treat two different competitive landscapes as one. Splitting the analysis or adding a tournament type flag would address this.
- Median imputation was applied before the train/test split, so test values influenced the fill. Fitting the imputation on training data only would be correct.
- The test set is roughly 118 rows, so small differences between models are within noise.
- The train/test split is random rather than time based, so the model can train on later tournaments and be tested on earlier ones. Splitting by year would better reflect how the model would actually be used.

## Running it

Requires Python 3 with jupyter, pandas, numpy, matplotlib, seaborn and scikit-learn.

Clone the repository and open `WorldCup_Analysis_Classification.ipynb` in VS Code or Jupyter Notebook.