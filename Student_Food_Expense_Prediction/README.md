# Student Daily Food Expense Prediction System

A Python desktop application that analyzes students' daily food expenses using **linear algebra, Principal Component Analysis (PCA), and regression modeling**. The project uses students' allowance and spending-habit data to build and evaluate models that estimate daily food expenses.

> **Project purpose:** This is an academic decision-support and learning tool. Predictions are estimates based on the available dataset; they are not guaranteed or exact future expenses.

## Project objectives

- Organize and validate the collected student food-expense dataset.
- Explore the dataset using descriptive statistics and visualizations.
- Represent the input features and target values using matrices and vectors.
- Apply linear algebra concepts such as matrix operations, rank, determinants, eigenvalues, and eigenvectors.
- Apply PCA to reduce the number of input features while measuring explained variance.
- Train and compare regression models using the original features and PCA-reduced features.
- Evaluate model performance using RMSE and R².
- Allow users to enter student information and receive an estimated daily food expense.

## Input and target variables

| Variable | Role | Description |
|---|---|---|
| `DailyAllowance_PHP` | Input feature | Student's daily allowance in Philippine pesos |
| `MealsPurchased` | Input feature | Number of meals purchased |
| `SnacksPurchased` | Input feature | Number of snacks purchased |
| `CampusHours` | Input feature | Number of hours spent on campus |
| `DailyFoodExpense_PHP` | Target | Recorded daily food expense in Philippine pesos |

The model-ready dataset is expected to contain these columns, along with `RespondentID` if available. If survey answers were collected in ranges and converted to representative numeric values, those values should be treated as approximations.

## Project structure

```text
StudentFoodExpensePrediction/
│
├── main.py
│
├── data/
│   ├── Student_Daily_Food_Expenses_Model_Ready.csv
│   └── original_responses.csv
│
├── modules/
│   ├── dataset_manager.py
│   ├── data_exploration.py
│   ├── preprocessing.py
│   ├── matrix_analysis.py
│   ├── vector_space.py
│   ├── transformations.py
│   ├── determinant_analysis.py
│   ├── eigen_pca.py
│   ├── regression.py
│   ├── model_evaluation.py
│   ├── prediction.py
│   ├── visualization.py
│   └── report_export.py
│
├── outputs/
│   ├── graphs/
│   ├── tables/
│   └── predictions/
│
├── notebooks/
│   └── linear_algebra_experiments.ipynb
│
├── tests/
│   ├── test_dataset.py
│   ├── test_matrix.py
│   └── test_prediction.py
│
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.10 or newer is recommended.
- Tkinter is used for the desktop interface. It is included with most standard Windows Python installations.
- Install the third-party libraries listed in `requirements.txt`.

### Install dependencies

Open a terminal in the project root folder and run:

```bash
python -m pip install -r requirements.txt
```

If `python` is not recognized on Windows, try:

```bash
py -m pip install -r requirements.txt
```

## Dataset setup

1. Place the cleaned, model-ready CSV file in the `data/` folder.
2. Use this expected filename unless the dataset-loading code is configured for another path:

   `data/Student_Daily_Food_Expenses_Model_Ready.csv`

3. Keep the original survey export separately as:

   `data/original_responses.csv`

4. Do not overwrite the original responses when cleaning or transforming the data.
5. Confirm that the model-ready CSV contains the required input and target columns listed above.

Do not upload personal or identifiable respondent information to a public repository. Remove names, email addresses, phone numbers, student numbers, and other identifying details before sharing the dataset. If the dataset cannot be shared publicly, keep it out of GitHub and provide it to authorized group members separately.

## Run the application

From the project root folder, run:

```bash
python main.py
```

On Windows, this can also be run with:

```bash
py main.py
```

The application uses Tkinter for its desktop interface. The dashboard and navigation are the starting interface; individual analysis and prediction pages must be connected to their corresponding modules as development progresses.

## Main system components

| Component | Responsibility |
|---|---|
| Dataset Management | Load, inspect, and validate the dataset |
| Data Exploration | Generate descriptive statistics and explore patterns |
| Preprocessing | Prepare features, split data, and standardize using training data |
| Linear Algebra | Perform matrix operations, rank, vector-space, and determinant analysis |
| Eigenvalues and PCA | Calculate eigenvalues/eigenvectors and reduce feature dimensions |
| Regression | Train models using original and PCA-reduced features |
| Model Evaluation | Compare predictions using RMSE and R² |
| Prediction | Estimate daily food expense for a new set of student inputs |
| Visualization and Report Export | Display graphs and save supported outputs |

## Model development and evaluation

The project compares two regression approaches:

1. **Full-feature regression:** trains using the original input features.
2. **PCA-based regression:** trains using principal components selected from the training data.

The dataset should be split into training and testing sets before fitting preprocessing steps. The scaler and PCA transformation must be fitted using the training data only, then applied to the test data. This helps prevent data leakage.

- **RMSE (Root Mean Squared Error):** measures the typical size of prediction errors in the target's units (PHP). Lower is generally better.
- **R² (Coefficient of Determination):** describes how much variation in the target is explained by the model. A higher value is generally better, but a negative test R² is possible and indicates performance worse than predicting the test-set mean.

Model comparison should use the same train/test split and evaluation data for a fair comparison. Do not claim that a model is accurate or reliable without reporting its measured test results and limitations.

## Team task distribution

The project is developed by five members:

| Member | Main responsibility | Assigned files |
|---|---|---|
| Member 1 | Main program, documentation, and integration | `main.py`, `README.md`, `requirements.txt` |
| Member 2 | Dataset management and exploration | `modules/dataset_manager.py`, `modules/data_exploration.py` |
| Member 3 | Linear algebra and preprocessing | `modules/preprocessing.py`, `modules/matrix_analysis.py`, `modules/vector_space.py`, `modules/transformations.py`, `modules/determinant_analysis.py` |
| Member 4 | Eigenvalues, PCA, and visualization | `modules/eigen_pca.py`, `modules/visualization.py`, `notebooks/linear_algebra_experiments.ipynb` |
| Member 5 | Regression, evaluation, prediction, and report export | `modules/regression.py`, `modules/model_evaluation.py`, `modules/prediction.py`, `modules/report_export.py` |

### Contribution guidelines

- Work on the files assigned to your team role.
- Keep function and class names consistent with the agreed project interfaces.
- Do not rename shared files or change function parameters without informing the integration lead.
- Test your module before submitting it.
- Explain any new dependency and add it to `requirements.txt` after discussing it with the team.
- The integration lead combines the completed modules and checks that the full application runs.

## Run tests

After the test files have been implemented, run the test suite from the project root:

```bash
python -m unittest discover -s tests -v
```

If the team chooses `pytest` instead, add it to `requirements.txt` and document the agreed test command.

## Limitations and responsible use

- Results depend on the quality, size, and representativeness of the collected survey data.
- Survey responses and midpoint conversions may contain recall error or approximation.
- The model may not represent students with spending patterns that are uncommon in the dataset.
- A prediction is an estimate, not financial advice or a guaranteed amount.
- Do not use the system to make high-stakes decisions about individual students.
- Report the dataset size, preprocessing choices, test metrics, and known limitations honestly.

## License

This repository is intended for academic project use. Add a license only if the project team has agreed on the terms for reuse and distribution.
