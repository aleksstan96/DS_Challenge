import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import aggs

def transform_date(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    df.loc[:, 'year'] = df[date_column].astype('datetime64[ns]').dt.year
    df.loc[:, 'week'] = df[date_column].astype('datetime64[ns]').dt.isocalendar().week
    df.loc[:, 'month'] = df[date_column].astype('datetime64[ns]').dt.month
    df.loc[:, 'dayofweek'] = df[date_column].astype('datetime64[ns]').dt.dayofweek
    df.loc[:, 'dayofmonth'] = df[date_column].astype('datetime64[ns]').dt.day
    df.loc[:, 'weekend'] = (df[date_column].astype('datetime64[ns]').dt.dayofweek >= 5).astype(int)
    df.drop(columns=[date_column], inplace=True)
    return df

def create_agg_features(df: pd.DataFrame) -> pd.DataFrame:
    agg_df = df.groupby(level=0).agg(aggs)
    agg_df.columns = [f"{col[0]}_{col[1]}" if col[1] != '<lambda_0>' else f"{col[0]}_mode" 
                    for col in agg_df.columns]
    agg_df = agg_df.reset_index()
    agg_df.set_index('user_id', inplace=True)
    agg_df.head()
    
    return agg_df

def main():
    registration_train = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleaned_data', 'registration_data_training_labeled.csv'), index_col=0)
    prev_lives_train = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleaned_data', 'previous_lives_training_data_labeled.csv'), index_col=0)
    ########## Training data ##########
    # transform date for registration data
    registration_train = transform_date(registration_train, 'registration_time_utc')
    registration_train.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleaned_data', 'registration_data_training_labeled_transformed.csv'))
    # transform date for previous lives data
    prev_lives_train = transform_date(prev_lives_train, 'registration_date')
    prev_lives_train.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleaned_data', 'previous_lives_training_data_labeled_transformed.csv'))
    # aggregate previous lives data
    agg_df = create_agg_features(prev_lives_train)

    # merge registration and aggregated previous lives data
    merged_df = pd.merge(registration_train, agg_df, left_index=True, right_index=True)
    merged_df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleaned_data', 'aggregated_data_training.csv'))
    
    ########## Test data ##########
    registration_test = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleaned_data', 'registration_data_test_labeled.csv'), index_col=0)
    prev_lives_test = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleaned_data', 'previous_lives_test_data_labeled.csv'), index_col=0)

    # transform date for registration data
    registration_test = transform_date(registration_test, 'registration_time_utc')
    registration_test.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleaned_data', 'registration_data_test_labeled_transformed.csv'))
    # transform date for previous lives data
    prev_lives_test = transform_date(prev_lives_test, 'registration_date')  
    prev_lives_test.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleaned_data', 'previous_lives_test_data_labeled_transformed.csv'))
    # aggregate previous lives data
    agg_df = create_agg_features(prev_lives_test)
    print(agg_df.head())

    # merge registration and aggregated previous lives data
    merged_df = pd.merge(registration_test, agg_df, left_index=True, right_index=True)
    merged_df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'cleaned_data', 'aggregated_data_test.csv'))    

if __name__ == "__main__":
    main()