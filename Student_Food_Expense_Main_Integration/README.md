# Student Daily Food Expense Prediction

A Python desktop application that estimates college students' daily food expenses
using daily allowance and spending habits. The project applies linear algebra,
Principal Component Analysis (PCA), and multiple linear regression. It compares
a full-feature regression model with a PCA-reduced regression model.

## Features

- Load and validate the model-ready CSV dataset.
- Use a reproducible 80:20 train/test split configured by `dataset_manager.py`.
- Standardize predictors using training-set statistics only.
- Train a full-feature multiple linear regression model.
- Perform PCA using the standardized training data only.
- Train and evaluate a PCA-reduced regression model.
- Compare test RMSE (PHP) and test R-squared.
- Predict daily food expense for a new student input.

## Required project structure

The current repository places `dataset_manager.py` in the project root and the
other model modules inside `modules/`. Keep this structure unless all imports
and paths are updated together.

```text
Student_Food_Expense_Prediction/
├── main.py
├── dataset_manager.py
├── data_exploration.py
├── README.md
├── requirements.txt
├── data/
│   └── Student_Daily_Food_Expenses_Model_Ready.csv
└── modules/
    ├── __init__.py
    ├── modules_regression.py
    ├── modules_model_evaluation.py
    ├── modules_prediction.py
    └── modules_report_export.py
```

The raw survey export may also be stored as `data/original_responses.csv`.

## Installation

Use Python 3.10 or newer.

1. Open a terminal in the `Student_Food_Expense_Prediction` project folder.
2. Create and activate a virtual environment if desired.
3. Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Tkinter is included with many standard Python installations. On some Linux
systems it may need to be installed through the operating system package manager.

## Dataset requirements

Place the cleaned dataset at:

`data/Student_Daily_Food_Expenses_Model_Ready.csv`

The CSV must contain these columns:

| Column | Meaning |
|---|---|
| `DailyAllowance_PHP` | Student's daily allowance in Philippine pesos |
| `MealsPurchased` | Number of meals purchased |
| `SnacksPurchased` | Number of snacks purchased |
| `CampusHours` | Hours spent on campus |
| `DailyFoodExpense_PHP` | Actual reported daily food expense in Philippine pesos |

`RespondentID` may also be present. Do not include names, email addresses, or
other unnecessary personal information in the modeling dataset.

The application does not create survey responses. It requires the actual
researcher-collected dataset. Do not report a sample count or model metric until
it has been computed from the verified CSV.

## Run the application

From the project root:

```bash
python main.py
```

1. Open the Dashboard tab.
2. Click **Load Dataset and Run Models**.
3. Open Model Analysis to review test RMSE and R-squared for both models.
4. Open Prediction, enter a new student's four predictor values, choose a model,
   and click **Predict Expense**.

## Method notes

- The full model uses all four standardized predictors.
- PCA is fitted on the standardized training matrix only.
- The number of principal components is selected to retain at least 95% of
  training-set variance, subject to the available feature count.
- The test set is used for final evaluation, not for fitting the standardizer
  or PCA basis.
- RMSE is reported in Philippine pesos because the target is in PHP.
- R-squared may be negative when predictions perform worse than predicting the
  test-set mean.

## Troubleshooting

- **Dataset not found:** confirm the cleaned CSV is inside the `data` folder
  and the filename matches exactly.
- **ModuleNotFoundError:** confirm the `modules` folder contains
  `__init__.py` and the module filenames match the imports in `main.py`.
- **Missing columns:** check the CSV headers against the required dataset
  columns listed above.
- **No GUI window:** confirm Tkinter is installed and run the program from a
  desktop environment.
- **Regression module issue:** the regression function should return a
  coefficient vector with an intercept and should accept standardized feature
  matrices.
- **R-squared module issue:** `calculate_r2()` should return the numeric `r2`
  value, not the function object. The main application includes a temporary
  fallback for the currently shared implementation.

## Dependencies

See `requirements.txt`.
