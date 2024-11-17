import xgboost as xgb
from sklearn import metrics
import numpy as np
import pandas as pd
from itertools import product

def round_and_clip(x):
    return np.clip(np.round(x), 0, 28)

def cross_validate_two_stage_model(df, features, target_col="days_active_first_28_days_after_registration", 
                                 fold_col="kfold", reg_model=None, n_folds=5, verbose=True, clf_models=None):
    
    # Default regression model
    if reg_model is None:
        reg_model = xgb.XGBRegressor(objective='reg:absoluteerror')
    
    results = {
        'train_mae': [],
        'valid_mae': []
    }
    
    for fold_ in range(n_folds):
        df_train = df[df[fold_col] != fold_].reset_index(drop=True)
        df_valid = df[df[fold_col] == fold_].reset_index(drop=True)
        
        x_train = df_train[features].values
        y_train = df_train[target_col].values
        x_valid = df_valid[features].values
        y_valid = df_valid[target_col].values
        
        clf_model = clf_models[fold_]
        train_pred_proba = clf_model.predict_proba(x_train)[:, 1]
        valid_pred_proba = clf_model.predict_proba(x_valid)[:, 1]
        
        threshold = 0.5
        train_pos_mask = train_pred_proba > threshold
        valid_pos_mask = valid_pred_proba > threshold
        
        # Stage 2: Regression only on predicted positive samples
        reg_model.fit(x_train[train_pos_mask], y_train[train_pos_mask])
        
        train_final = np.zeros_like(y_train, dtype=float)
        valid_final = np.zeros_like(y_valid, dtype=float)
        
        if train_pos_mask.any():
            train_final[train_pos_mask] = reg_model.predict(x_train[train_pos_mask])
        if valid_pos_mask.any():
            valid_final[valid_pos_mask] = reg_model.predict(x_valid[valid_pos_mask])
        
        train_final = round_and_clip(train_final)
        valid_final = round_and_clip(valid_final)
        
        train_mae = metrics.mean_absolute_error(y_train, train_final)
        valid_mae = metrics.mean_absolute_error(y_valid, valid_final)
        
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

def tune_two_stage_model(df, features, param_grid, target_col="days_active_first_28_days_after_registration", n_folds=5, **cv_kwargs):
    # Train binary classifiers for each fold - stage 1
    clf_models = {}
    for fold_ in range(n_folds):
        clf_model = xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, max_depth=6)
        
        df_train = df[df['kfold'] != fold_].reset_index(drop=True)
        X_train = df_train[features].values
        y_train_binary = (df_train[target_col].values > 0).astype(int)
        
        clf_model.fit(X_train, y_train_binary)
        clf_model.fit(X_train, y_train_binary)
        clf_models[fold_] = clf_model
    
    cv_kwargs['clf_models'] = clf_models
    
    best_score = float('inf')
    best_params = None
    all_results = []
    
    param_keys = param_grid.keys()
    param_values = param_grid.values()
    
    for params in product(*param_values):
        current_params = dict(zip(param_keys, params))
        reg_model = xgb.XGBRegressor(**current_params)
        
        print(f"\nTrying parameters: {current_params}")
        results = cross_validate_two_stage_model(df, features, reg_model=reg_model, **cv_kwargs)
        
        mean_valid_mae = np.mean(results['valid_mae'])
        all_results.append({
            'params': current_params,
            'mean_valid_mae': mean_valid_mae,
            'std_valid_mae': np.std(results['valid_mae'])
        })
        
        if mean_valid_mae < best_score:
            best_score = mean_valid_mae
            best_params = current_params
    
    # Print results
    all_results.sort(key=lambda x: x['mean_valid_mae'])
    print("\n=== All Results (sorted by validation MAE) ===")
    for result in all_results:
        print(f"\nParameters: {result['params']}")
        print(f"Valid MAE: {result['mean_valid_mae']:.4f} ± {result['std_valid_mae']:.4f}")
    
    print(f"\n=== Best Parameters ===")
    print(f"Parameters: {best_params}")
    print(f"Valid MAE: {best_score:.4f}")

    # Save results to csv
    pd.DataFrame(all_results).to_csv("tuning_results_two_stage.csv", index=False)
    
    return best_params, all_results

if __name__ == "__main__":
    df = pd.read_csv("/root/projects/ds_chlg/Data Science Challenge/Data Science/folds/train_folds_merged_5folds.csv")
    features = [f for f in df.columns if f not in ("kfold", "days_active_first_28_days_after_registration")]
    
    param_grid = {
        'objective': ['reg:absoluteerror'],
        'eta': [ 0.08, 0.1],
        'gamma': [0.5],
        'max_depth': [6, 9],
        'subsample': [0.8, 1.0],
        'min_child_weight': [3, 5]
    }
    
    best_params, results = tune_two_stage_model(df, features, param_grid, verbose=False)