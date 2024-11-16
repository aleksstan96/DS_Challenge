import pandas as pd
import numpy as np
import sys
import os
from sklearn.preprocessing import LabelEncoder
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import nominal_features_registration, nominal_features_prev_lives

def label_encode_column(dataframes, features):
    # concatenated dataframe
    for i, df in enumerate(dataframes):
        df['source'] = f'df_{i}'
    df_concatenated = pd.concat(dataframes)

    for feature in features:
        print(f"Label encoding feature: {feature}")
        le = LabelEncoder()
        
        if df_concatenated[feature].dtype == 'bool':
            df_concatenated.loc[:, feature] = df_concatenated[feature].astype(int)
            for i, df in enumerate(dataframes):
                df.loc[:, feature] = df_concatenated[df_concatenated['source'] == f'df_{i}'][feature].values
        else:
            df_concatenated.loc[:, feature] = le.fit_transform(df_concatenated[feature])
            for i, df in enumerate(dataframes):
                df.loc[:, feature] = df_concatenated[df_concatenated['source'] == f'df_{i}'][feature].values
        
        # print(df[feature].value_counts())

    for df in dataframes:
        df.drop(columns=['source'], inplace=True)
    return dataframes

def main():
    registration_train = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleaned_data', 'registration_data_training.csv'))
    registration_test = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleaned_data', 'registration_data_test.csv'))
    prev_lives_train = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleaned_data', 'previous_lives_training_data.csv'))
    prev_lives_test = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleaned_data', 'previous_lives_test_data.csv'))

    # handling features that are present in both registration and previous lives
    # intersection of nominal features for registration and previous lives
    print("=========== Label encoding both registration and previous lives ===========")
    intersection_features = np.intersect1d(nominal_features_registration, nominal_features_prev_lives)
    transformed_dataframes = label_encode_column([registration_train, registration_test, prev_lives_train, prev_lives_test], intersection_features)

    print('=========== Label encoding registration ============')
    transformed_dataframes1 = label_encode_column([transformed_dataframes[0], transformed_dataframes[1]], np.setdiff1d(nominal_features_registration, intersection_features))
    transformed_dataframes1[0].to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleaned_data', 'registration_data_training_labeled.csv'), index=False)
    transformed_dataframes1[1].to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleaned_data', 'registration_data_test_labeled.csv'), index=False)

    print("=========== Label encoding previous lives ===========")
    transformed_dataframes2 = label_encode_column([transformed_dataframes[2], transformed_dataframes[3]], np.setdiff1d(nominal_features_prev_lives, intersection_features))
    transformed_dataframes2[0].to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleaned_data', 'previous_lives_training_data_labeled.csv'), index=False)
    transformed_dataframes2[1].to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleaned_data', 'previous_lives_test_data_labeled.csv'), index=False)
    
if __name__ == '__main__':
    main()