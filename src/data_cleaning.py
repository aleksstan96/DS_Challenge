import pandas as pd
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import country_threshold, manufacturer_threshold

def replace_nan_values(df, value):
    for feature in df.columns:
        df[feature] = df[feature].fillna(value)
    return df

def clean_column(df, df_concatenated, feature, threshold):
    feature_counts = df_concatenated[feature].value_counts(normalize=True)
    valid_features = feature_counts[feature_counts >= threshold].index
    
    df[feature] = df[feature].apply(lambda x: 'Other' if x not in valid_features else x)
    return df

def main():
    # read data
    registration_train = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'registration_data_training.csv'))
    registration_test = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'registration_data_test.csv'))
    prev_lives_train = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'previous_lives_training_data.csv'))
    prev_lives_test = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'previous_lives_test_data.csv'))
    # concatenated dataframe
    df_concatenated = pd.concat([registration_train, registration_test, prev_lives_train, prev_lives_test])

    # specific: mapping both "INFINIX" and "INFINIX MOBILITY LIMITED" to "INFINIX" for registration_device_manufacturer column
    df_concatenated['registration_device_manufacturer'] = df_concatenated['registration_device_manufacturer'].apply(lambda x: 'INFINIX' if x in ['INFINIX', 'INFINIX MOBILITY LIMITED'] else x)
    
    # registration data - training
    registration_train = replace_nan_values(registration_train, 'NaN')
    registration_train = clean_column(registration_train, df_concatenated, 'registration_country', country_threshold)
    registration_train = clean_column(registration_train, df_concatenated, 'registration_device_manufacturer', manufacturer_threshold)
    registration_train.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleaned_data', 'registration_data_training.csv'), index=False)

    # registration data - test
    registration_test = replace_nan_values(registration_test, 'NaN')
    registration_test = clean_column(registration_test, df_concatenated, 'registration_country', country_threshold)
    registration_test = clean_column(registration_test, df_concatenated, 'registration_device_manufacturer', manufacturer_threshold)
    registration_test.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleaned_data', 'registration_data_test.csv'), index=False)

    # prev_lives data - training
    prev_lives_train = replace_nan_values(prev_lives_train, 'NaN')
    prev_lives_train = clean_column(prev_lives_train, df_concatenated, 'registration_country', country_threshold)
    prev_lives_train.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleaned_data', 'previous_lives_training_data.csv'), index=False)

    # prev_lives data - test
    prev_lives_test = replace_nan_values(prev_lives_test, 'NaN')
    prev_lives_test = clean_column(prev_lives_test, df_concatenated, 'registration_country', country_threshold)
    prev_lives_test.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleaned_data', 'previous_lives_test_data.csv'), index=False)


if __name__ == '__main__':
    main()