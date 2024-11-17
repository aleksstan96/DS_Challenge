# User Activity Prediction - Data Science Challenge 🕹️

This repository contains the solution for the Nordeus Data Science Challenge focused on predicting the number of active days a user will have within the first 28 days after re-registering for a game. The project involves data preparation, exploratory analysis, feature engineering, model building, and evaluation.

## Project Overview 🎯

### Goal
To predict the target variable representing the number of days a user was active in the first 28 days after re-registering.

### Datasets
The data consists of two primary datasets:

#### Registration Data
- Captures user activity on the day of re-registration
- Includes features like device information, platform, session statistics, and gameplay metrics

#### Previous Lives Data
- Contains historical activity from the user's past gameplay sessions
- Includes metrics like lifetime active days, purchases, and previous registration details

## Methodology 🛠️

*Note: Detailed explanation of the modeling process is in the Jupyter notebook: `notebooks/eda.ipynb`.*

### 1. Data Cleaning and Preprocessing
**Purpose**: Address inconsistencies and prepare datasets for merging and modeling.

**Steps**:
- Handling rare categories in nominal features (e.g., grouping rare registration_country values into "Other")
- Standardizing date and time features for consistency
- Imputing missing values and removing outliers where applicable

### 2. Exploratory Data Analysis (EDA)
**Insights**:
- The target variable has a skewed and zero-inflated distribution, with many users inactive (0 days). It also has a U-shape, indicating two groups of users:
  - Users who are active for only a few days and then become inactive
  - Users who are active for more than ~20 days
- Nominal features like registration_country exhibit many rare values

### 3. Feature Engineering
**Objective**: Enhance predictive power by creating derived features. Feature engineering is done on both previous_lives data and the merged data.

**Key Features**:
- **Recency, Frequency, Age**:
  - Time since last activity
  - Frequency of past activity (e.g., days_active_lifetime_mean)
  - Lifetime activity span (registration_date_min to registration_date_max)
- **Aggregations**:
  - Unique counts (nunique) for temporal features like year, month, week
  - Summarized statistics (sum, mean, std) for user activity and transactions
- **Behavioral Ratios**:
  - Match win rate, spending intensity, etc.

### 4. Model Building
1. **First Approach**: Plain XGBoostRegressor
2. **Second Approach**: Two-Stage Modeling:
   - **Stage 1** (Binary Classification):
     - Predict whether a user will be active (days > 0) or inactive (days = 0)
     - Model: XGBoostClassifier
   - **Stage 2** (Regression):
     - Predict the number of active days for users classified as active
     - Model: XGBoostRegressor

### 5. Evaluation
- **Metric**: Mean Absolute Error (MAE)
- **Validation**: K-fold cross-validation with K=5 folds

### 6. Hyperparameter Tuning
Grid search over key parameters:
- Learning rate (eta)
- Maximum depth (max_depth)
- Regularization terms (reg_lambda, reg_alpha)
- Tree-specific parameters

## Results 📊

### First Approach
**Parameters**:
- {'objective': 'reg:absoluteerror', 'eta': 0.08, 'gamma': 0.5, 'max_depth': 6, 'reg_lambda': 0.01, 'min_child_weight': 3}
Validation MAE: 5.2774

### Second Approach
**Parameters**:
- For XGBoostRegressor: {'objective': 'reg:absoluteerror', 'eta': 0.08, 'gamma': 0.5, 'max_depth': 6, 'subsample': 1.0, 'min_child_weight': 5}
- For XGBoostClassifier: default parameters, objective='binary:logistic'
Validation MAE: 5.3551

## Ideas for Future Work 💡
- Since the distribution of the target variable is U-shaped, it suggests there are two groups of users. This information can be leveraged in the modeling process, e.g., by using mixture models.
- Explore ensemble methods to combine predictions for improved accuracy.
