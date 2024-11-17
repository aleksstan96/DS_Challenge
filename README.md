User Activity Prediction - Data Science Challenge
This repository contains the solution for Nordeus Data Science Challenge focused on predicting the number of active days a user will have within the first 28 days after re-registering for a game. The project involves data preparation, exploratory analysis, feature engineering, model building, and evaluation.

Project Overview
Goal
To predict the target variable representing the number of days a user was active in the first 28 days after re-registering.

Datasets
The data consists of two primary datasets:

Registration Data:
- Captures user activity on the day of re-registration.
-Includes features like device information, platform, session statistics, and gameplay metrics.
Previous Lives Data:
- Contains historical activity from the user’s past gameplay sessions.
- Includes metrics like lifetime active days, purchases, and previous registration details.

Methodology
Note: Detailed explanation of the modeling process is in the Jupyter notebook notebooks/eda.ipynb.

1. Data Cleaning and Preprocessing
Purpose: Address inconsistencies and prepare datasets for merging and modeling.
Steps:
- Handling rare categories in nominal features (e.g., grouping rare registration_country values into "Other").
- Standardizing date and time features for consistency.
- Imputing missing values and removing outliers where applicable.
2. Exploratory Data Analysis (EDA)
Insights:
- The target variable has a skewed and zero-inflated distribution, with many users inactive (0 days). Also, it has a U-shape, indicating there are two groups of users:
  - Users who are active for a few days and then become inactive.
  - Users who are active for more than ~20 days.
- Nominal features like registration_country exhibit many rare values.
3. Feature Engineering
Objective: Enhance predictive power by creating derived features. Feature engineering is done on both previous_lives data as well as merged data.
Key Features:
- Recency, Frequency, Age:
  - Time since last activity.
  - Frequency of past activity (e.g., days_active_lifetime_mean).
  - Lifetime activity span (registration_date_min to registration_date_max).
- Aggregations:
  - Unique counts (nunique) for temporal features like year, month, week.
Summarized statistics (sum, mean, std) for user activity and transactions.
- Behavioral Ratios:
  - Match win rate, spending intensity, etc.
4. Model Building
First approach: Plain XGBoostRegressor.
Second approach: Two-Stage Modeling:
- Stage 1 (Binary Classification): Predict whether a user will be active (days > 0) or inactive (days = 0). Model: XGBoostClassifier.
- Stage 2 (Regression): Predict the number of active days for users classified as active. Model: XGBoostRegressor.
5. Evaluation:
Metric: Mean Absolute Error (MAE).
K-fold cross-validation with K=5 folds.
6. Hyperparameter Tuning
Grid search over key parameters:
- Learning rate (eta), maximum depth (max_depth), regularization terms (reg_lambda, reg_alpha), and tree-specific parameters.

Ideas for future work (not implemented):
- Since distribution of target variable is U-shaped, it tells us there are two groups of users. We can use this information in the modeling process, e.g. by using mixture models.
- Explore ensemble methods to combine predictions.