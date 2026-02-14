# SHAP Projects Repository

This repository contains a practical SHAP learning path:

1. Introductory notebooks to build intuition for SHAP values and background data.
2. Four end-to-end case studies:
   - Classical tabular classification
   - Classical tabular regression
   - PyTorch ANN classification
   - TensorFlow/Keras CNN image classification

The goal is to move from "what SHAP is" to "how to apply it correctly across model types".

## Repository Structure

```text
SHAP-Projects/
|- SHAP Introduction/
|  |- shap_intro.ipynb
|  |- shap_background_data.ipynb
|- SHAP Classification Case Study/
|  |- classification_shap.ipynb
|  |- Churn_Modelling.csv
|- SHAP Regression Case Study/
|  |- regression_shap.ipynb
|- SHAP Pytorch ANN/
|  |- ann_shap.ipynb
|  |- loan_data.csv
|- SHAP TensorFlow ANN/
|  |- cnn_shap_MNIST.ipynb
|- pyproject.toml
|- uv.lock
|- main.py
```

## Environment and Dependencies

The project is configured for Python 3.11+ via `pyproject.toml` and uses these major libraries:

- `shap`
- `scikit-learn`
- `xgboost`
- `lightgbm`
- `optuna`
- `tensorflow` / `keras`
- `torch`
- `feature-engine`
- `pandas`, `numpy`, `matplotlib`, `seaborn`

### Setup

```bash
uv sync
```

or with pip:

```bash
pip install -e .
```

For notebook execution:

```bash
pip install ipykernel
```

## Learning Flow and Case Studies

## 1) SHAP Introduction

### A. `SHAP Introduction/shap_intro.ipynb`

Purpose:
- Introduce SHAP on a simple multiclass tabular dataset (Iris).

Workflow:
1. Load Iris using `shap.datasets.iris(display=True)`.
2. Train/test split with stratification.
3. Encode target labels with `LabelEncoder`.
4. Train `RandomForestClassifier`.
5. Evaluate train/test accuracy.
6. Create `shap.TreeExplainer` with `model_output="probability"`.
7. Explain a single test row and generate waterfall plots per class.

SHAP focus:
- Understand base value vs per-feature contribution.
- See class-wise explanations in a multiclass setting.

### B. `SHAP Introduction/shap_background_data.ipynb`

Purpose:
- Demonstrate how SHAP explanations change with different background data.

Workflow:
1. Build synthetic dataset with two binary features:
   - `knows_python`
   - `knows_genai`
2. Create target `placement` from feature logic (`df.any(axis=1).astype(int)`).
3. Train/test split and fit `GradientBoostingClassifier`.
4. Build `TreeExplainer` and explain selected rows.
5. Repeat explanations with three backgrounds:
   - All training rows
   - Positive-class rows only
   - Negative-class rows only
6. Compare base values and SHAP values across the same rows.

SHAP focus:
- Background data defines the reference expectation.
- Same model + same row can produce different attribution values when baseline changes.

## 2) Tabular Classification Case Study

Notebook: `SHAP Classification Case Study/classification_shap.ipynb`  
Dataset: `SHAP Classification Case Study/Churn_Modelling.csv`

Objective:
- Predict customer churn (`exited`) and explain predictions globally and locally.

Pipeline summary:
1. Load data and drop first three identifier columns.
2. Normalize column names to lowercase.
3. Define features and target:
   - `X = df.drop(columns=["exited"])`
   - `y = df["exited"]`
4. EDA:
   - Histogram + boxplot for numeric variables
   - Countplot and frequency checks for categorical variables
5. Preprocessing with `ColumnTransformer`:
   - `MinMaxScaler` on numeric columns
   - `OneHotEncoder(handle_unknown="ignore", sparse_output=False)` on categorical columns
   - passthrough for remainder columns
6. Model:
   - Soft-voting ensemble with:
     - `RandomForestClassifier(class_weight="balanced")`
     - `XGBClassifier(class_weight="balanced")`
     - `LGBMClassifier(class_weight="balanced")`
7. Hyperparameter tuning:
   - `Optuna` with 100 trials
   - 5-fold stratified CV
   - Optimization target: mean recall
8. Refit final tuned model and evaluate with `classification_report`.
9. SHAP:
   - Model-agnostic explainer:
     - `prediction_function = model.predict_proba(X)[:,1]`
     - `shap.Explainer(model=prediction_function, masker=X_train, link=identity)`
   - Global plots on sampled rows:
     - bar, beeswarm, violin, heatmap
   - Local plots:
     - waterfall, local bar, force, decision
   - Batch analysis:
     - decision plot for multiple rows

What this case study teaches:
- How to connect a full ML pipeline (preprocess + ensemble + tuning) with SHAP.
- Differences between global importance views and per-customer explanations.

## 3) Tabular Regression Case Study

Notebook: `SHAP Regression Case Study/regression_shap.ipynb`

Objective:
- Predict California housing target and explain feature impact on continuous predictions.

Pipeline summary:
1. Load dataset via `fetch_california_housing(as_frame=True)`.
2. Train/test split.
3. Train baseline `XGBRegressor`.
4. Evaluate with:
   - RMSE (`root_mean_squared_error`)
   - R2 (`r2_score`)
5. Tune XGBoost with `Optuna` (50 trials, maximize test R2).
6. Train best model from study parameters.
7. SHAP analysis using `TreeExplainer(model=best_model, data=X_train)`.
8. Global interpretation:
   - bar plot
   - heatmap (including custom instance ordering)
   - beeswarm
   - dependence scatter plots for selected features (`MedInc`, `AveOccup`, `HouseAge`)
9. Local interpretation:
   - waterfall
   - force
   - local bar

What this case study teaches:
- SHAP usage for regression outputs.
- How dependence plots reveal non-linear and interaction-driven behavior.

## 4) PyTorch ANN Classification Case Study

Notebook: `SHAP Pytorch ANN/ann_shap.ipynb`  
Dataset: `SHAP Pytorch ANN/loan_data.csv`

Objective:
- Build a neural network for loan status prediction and interpret it with SHAP DeepExplainer.

Pipeline summary:
1. Load loan dataset and perform EDA for numeric/categorical columns.
2. Define features and target:
   - `X = df.drop(columns=["loan_status"])`
   - `y = df["loan_status"]`
3. Preprocessing with `ColumnTransformer`:
   - Numeric pipeline:
     - `Winsorizer` (IQR capping)
     - `RobustScaler`
   - Nominal categorical:
     - `OneHotEncoder(drop="first", handle_unknown="ignore")`
   - Ordered categorical (`person_education`):
     - `OrdinalEncoder` with explicit order:
       `High School < Associate < Bachelor < Master < Doctorate`
4. Train/test split with stratification and shuffle.
5. Build custom `LoanDataset` class and `DataLoader`s.
6. Build PyTorch model (`MyModel`):
   - Dense network with BatchNorm + ReLU blocks
   - Final sigmoid output for binary probability
7. Train with:
   - `Adam` optimizer
   - `BCELoss`
   - epoch-wise train/test loss tracking
8. Evaluate via thresholded predictions and `classification_report`.
9. SHAP:
   - Sample training rows as background tensor
   - `shap.DeepExplainer(model, background_data)`
   - Explain a batch of test rows
   - Repackage into `shap.Explanation` with transformed feature names
   - Plot:
     - global bar and beeswarm
     - local waterfall and bar
     - decision plot

What this case study teaches:
- SHAP on neural nets in PyTorch.
- Importance of correct background tensors and readable transformed feature names.

## 5) TensorFlow/Keras CNN (MNIST) Case Study

Notebook: `SHAP TensorFlow ANN/cnn_shap_MNIST.ipynb`

Objective:
- Classify MNIST digits with CNN and visualize pixel-level SHAP attributions.

Pipeline summary:
1. Load MNIST from `keras.datasets`.
2. Convert to `float32`, normalize pixel values to `[0,1]`.
3. Add channel dimension with `np.expand_dims(..., axis=3)`.
4. Build CNN:
   - Input layer
   - Conv2D(16, 3x3, ReLU)
   - MaxPool2D
   - Flatten
   - Dense(32, ReLU)
   - Dropout(0.4)
   - Dense(10, Softmax)
5. Compile with Adam + sparse categorical crossentropy.
6. Train for 10 epochs, batch size 32.
7. Evaluate on test set.
8. SHAP:
   - Use 200 training images as background
   - `shap.DeepExplainer(model, background)`
   - Explain first 5 test images
   - Build per-class SHAP value list
   - Visualize with `shap.plots.image(...)`

What this case study teaches:
- SHAP for image models and multiclass pixel attribution.
- Class-specific explanation maps for CNN predictions.

## Key SHAP Concepts Demonstrated Across the Repo

1. Explainer selection by model type:
   - Tree models: `TreeExplainer`
   - Neural nets: `DeepExplainer`
   - General callable models: `shap.Explainer`
2. Local vs global explanations:
   - Local: waterfall, force, local bar, single-row decision
   - Global: bar, beeswarm, violin, heatmap, dependence/scatter
3. Background data sensitivity:
   - Explicitly shown in `shap_background_data.ipynb`
4. Multiclass handling:
   - Iris and MNIST case studies show class-wise output interpretation.

## How to Use This Repo Effectively

Recommended order:

1. `SHAP Introduction/shap_intro.ipynb`
2. `SHAP Introduction/shap_background_data.ipynb`
3. `SHAP Classification Case Study/classification_shap.ipynb`
4. `SHAP Regression Case Study/regression_shap.ipynb`
5. `SHAP Pytorch ANN/ann_shap.ipynb`
6. `SHAP TensorFlow ANN/cnn_shap_MNIST.ipynb`

Execution tips:

- Run notebooks top-to-bottom to preserve intermediate variables.
- For reproducibility, keep the existing random seeds where defined.
- Deep SHAP can be memory-intensive; reduce batch sizes or background sample size if needed.

## Notes

- `README.md` now documents repository intent and each notebook pipeline in detail.
- `main.py` is minimal and not part of the notebook workflows.
