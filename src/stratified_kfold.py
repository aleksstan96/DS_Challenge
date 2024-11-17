from sklearn.model_selection import StratifiedKFold
import pandas as pd

def stratified_kfold(df, target_col, n_splits=5, dataset_name=""):
    df["kfold"] = -1
    # randomize the rows of the data
    df = df.sample(frac=1).reset_index(drop=True)

    y = df[target_col].values
    kf = StratifiedKFold(n_splits=n_splits)
    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f
    # save
    df.to_csv(f"/root/projects/ds_chlg/Data Science Challenge/Data Science/folds/train_folds_{dataset_name}_{n_splits}folds.csv", index=False)
    print(f"Saved {dataset_name} folds to /root/projects/ds_chlg/Data Science Challenge/Data Science/folds/train_folds_{dataset_name}_{n_splits}folds.csv")
    return df


if __name__ == "__main__":
    df = pd.read_csv('/root/projects/ds_chlg/Data Science Challenge/Data Science/aggregated_data/merged_data_training.csv')
    stratified_kfold(df, "days_active_first_28_days_after_registration", n_splits=2, dataset_name="merged")
