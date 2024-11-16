import xgboost as xgb
from sklearn import metrics
import numpy as np
import pandas as pd

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

if __name__ == "__main__":
    df = pd.read_csv("/root/projects/ds_chlg/Data Science Challenge/Data Science/folds/train_folds_agg_data_5folds.csv")
    features = [f for f in df.columns if f not in ("kfold", "days_active_first_28_days_after_registration")]
    
    # plain XGBoost regressor with mae objective
    print("========= XGBoost regressor with mae objective")
    results, valid_preds = cross_validate_model(df, features)

    # XGBoost regressor with Poisson objective
    print("========= XGBoost regressor with Poisson objective")
    model = xgb.XGBClassifier(
        objective='count:poisson',
    )
    results = cross_validate_model(df, features, model=model)

    # XGBoost regressor with tweedie objective
    print("========= XGBoost regressor with tweedie objective")
    model = xgb.XGBRegressor(
        objective='reg:tweedie',
    )
    results, valid_preds = cross_validate_model(df, features, model=model, prediction_transformer=round_and_clip)

    # Example with custom model and metrics
    # from sklearn.ensemble import RandomForestClassifier
    # custom_model = RandomForestClassifier(n_estimators=100, random_state=42)
    # custom_metrics = {
    #     'accuracy': metrics.accuracy_score,
    #     'mae': metrics.mean_absolute_error
    # }
    
    # results = cross_validate_model(
    #     df,
    #     features,
    #     model=custom_model,
    #     metrics_funcs=custom_metrics
    # )