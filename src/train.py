import xgboost as xgb
from sklearn import metrics
import numpy as np
import pandas as pd
from itertools import product

def round_and_clip(x):
    return np.clip(np.round(x), 0, 28)

def cross_validate_model(df, features, target_col = "days_active_first_28_days_after_registration", 
                         fold_col = "kfold", model=None, metrics_funcs = None, n_folds: int = 5,
                         prediction_transformer=None, verbose=True):

    # Default model
    if model is None:
        model = xgb.XGBRegressor(
            objective='reg:absoluteerror'
        )
        # since model is regression, we need to round predictions to integers and bound them between 0 and 28
        prediction_transformer = round_and_clip
    
    # If no transformer is specified, do nothing
    if prediction_transformer is None:
        prediction_transformer = lambda x: x

    # Default metric is MAE
    if metrics_funcs is None:
        metrics_funcs = {
            'mae': metrics.mean_absolute_error
        }
    
    results = {
        'train_mae': [],
        'valid_mae': []
    }
    
    # k-fold cross validation
    for fold_ in range(n_folds):
        df_train = df[df[fold_col] != fold_].reset_index(drop=True)
        df_valid = df[df[fold_col] == fold_].reset_index(drop=True)
        
        x_train = df_train[features].values
        y_train = df_train[target_col].values
        x_valid = df_valid[features].values
        y_valid = df_valid[target_col].values
        
        model.fit(x_train, y_train)
        
        train_preds = model.predict(x_train)
        valid_preds = model.predict(x_valid)
        
        # Apply transformation before calculating metrics
        train_preds_transformed = prediction_transformer(train_preds)
        valid_preds_transformed = prediction_transformer(valid_preds)
        
        train_mae = metrics.mean_absolute_error(y_train, train_preds_transformed)
        valid_mae = metrics.mean_absolute_error(y_valid, valid_preds_transformed)
        
        results['train_mae'].append(train_mae)
        results['valid_mae'].append(valid_mae)
        
        if verbose:
            print(f"Fold {fold_}, Train MAE = {train_mae:.4f}, Valid MAE = {valid_mae:.4f}")
    
    print("\nMean scores:")
    for metric_name, scores in results.items():
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        print(f"{metric_name}: {mean_score:.4f} ± {std_score:.4f}")
    
    return results

def tune_and_cross_validate(df, features, param_grid, base_model_class, **cv_kwargs):
    """
    Performs grid search with cross-validation
    """
    best_score = float('inf')
    best_params = None
    all_results = []
    
    # Generate all parameter combinations
    param_keys = param_grid.keys()
    param_values = param_grid.values()
    
    for params in product(*param_values):
        current_params = dict(zip(param_keys, params))
        
        # Create model with current parameters
        model = base_model_class(**current_params)
        
        print(f"\nTrying parameters: {current_params}")
        results = cross_validate_model(df, features, model=model, **cv_kwargs)
        
        # Store results along with parameters
        mean_valid_mae = np.mean(results['valid_mae'])
        all_results.append({
            'params': current_params,
            'mean_valid_mae': mean_valid_mae,
            'std_valid_mae': np.std(results['valid_mae'])
        })
        
        # Track best parameters
        if mean_valid_mae < best_score:
            best_score = mean_valid_mae
            best_params = current_params
    
    # Sort and print results
    all_results.sort(key=lambda x: x['mean_valid_mae'])
    print("\n=== All Results (sorted by validation MAE) ===")
    for result in all_results:
        print(f"\nParameters: {result['params']}")
        print(f"Valid MAE: {result['mean_valid_mae']:.4f} ± {result['std_valid_mae']:.4f}")
    
    print(f"\n=== Best Parameters ===")
    print(f"Parameters: {best_params}")
    print(f"Valid MAE: {best_score:.4f}")

    # Save results to csv
    pd.DataFrame(all_results).to_csv("tuning_results.csv", index=False)
    
    return best_params, all_results

if __name__ == "__main__":
    df = pd.read_csv("/root/projects/ds_chlg/Data Science Challenge/Data Science/folds/train_folds_merged_5folds.csv")
    features = [f for f in df.columns if f not in ("kfold", "days_active_first_28_days_after_registration")]
    
    # Example parameter grids for different objectives
    mae_param_grid = {
        'objective': ['reg:absoluteerror'],  # Most relevant objectives
        # 'objective': ['reg:absoluteerror', 'count:poisson', 'reg:tweedie'],  # Most relevant objectives
        'eta': [0.08, 0.1],                                   # Same as learning_rate
        # 'gamma': [0.5, 0.1, 0.3],
        'max_depth': [5, 6, 7],                                  # Tree depth
        'n_estimators': [100, 150],                          # Number of trees
        # 'subsample': [1.0],                             # Row sampling per tree
        # 'reg_lambda': [1.0, 10.0],                           # L2 regularization (most important reg parameter)
        'min_child_weight': [3, 5]                           # Controls overfitting
    }
    
    # Tune XGBoost with MAE objective
    print("\n========= Tuning XGBoost regressor with different objectives")
    best_params_mae, results_mae = tune_and_cross_validate(
        df, 
        features, 
        mae_param_grid, 
        xgb.XGBRegressor,
        prediction_transformer=round_and_clip,
        verbose=False
    )
    
    # Tune XGBoost with Poisson objective
    # print("\n========= Tuning XGBoost regressor with Poisson objective")
    # best_params_poisson, results_poisson = tune_and_cross_validate(
    #     df, 
    #     features, 
    #     poisson_param_grid, 
    #     xgb.XGBRegressor,
    #     prediction_transformer=round_and_clip
    # )