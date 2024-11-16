from sklearn.model_selection import StratifiedKFold

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
    df.to_csv(f"../folds/train_folds_{dataset_name}_{n_splits}folds.csv", index=False)
    print(f"Saved {dataset_name} folds to ../folds/train_folds_{dataset_name}_{n_splits}folds.csv")
    return df
