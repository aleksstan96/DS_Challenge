import pandas as pd
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import aggs

def transform_date(df, date_column, drop_original=True):
    df.loc[:, 'year'] = df[date_column].astype('datetime64[ns]').dt.year
    df.loc[:, 'week'] = df[date_column].astype('datetime64[ns]').dt.isocalendar().week
    df.loc[:, 'month'] = df[date_column].astype('datetime64[ns]').dt.month
    df.loc[:, 'dayofweek'] = df[date_column].astype('datetime64[ns]').dt.dayofweek
    df.loc[:, 'dayofmonth'] = df[date_column].astype('datetime64[ns]').dt.day
    df.loc[:, 'weekend'] = (df[date_column].astype('datetime64[ns]').dt.dayofweek >= 5).astype(int)
    if drop_original:
        df.drop(columns=[date_column], inplace=True)
    return df



def create_agg_features(df):
    agg_df = df.groupby(level=0).agg(aggs)
    agg_df.columns = [f"{col[0]}_{col[1]}" if col[1] != '<lambda_0>' else f"{col[0]}_mode" 
                    for col in agg_df.columns]
    agg_df = agg_df.reset_index()
    agg_df.set_index('user_id', inplace=True)
    # rename columns
    agg_df.rename(columns={'registration_date_min': 'first_registration_date_prev_lives', 'registration_date_max': 'last_registration_date_prev_lives'}, inplace=True)
    agg_df.head()
    
    return agg_df

def create_features_on_merged_data(df, drop_cols=[]):
    # recency
    df['recency_days'] = (pd.to_datetime(df['registration_time_utc']) - pd.to_datetime(df['last_registration_date_prev_lives'])).dt.days
    # age
    df['age_days'] = (pd.to_datetime(df['registration_time_utc']) - pd.to_datetime(df['first_registration_date_prev_lives'])).dt.days
    # frequency
    df['frequency'] = df['days_active_lifetime_sum'] / df['year_nunique']
    # match win rate
    df['match_win_rate'] = df['total_match_won_count'] / df['total_match_played_count']
    # avg tokens spent per session
    df['avg_tokens_spent_per_session'] = df['tokens_spent'] / df['session_count']
    # inactive days
    df['inactive_days'] = df['age_days'] - df['days_active_lifetime_sum']
    # registration frequency
    df['registration_frequency_days'] = df['age_days'] / df['year_nunique']
    # fill nan values for columns: days_active_lifetime_std, transaction_count_iap_lifetime_std,  match_win_rate
    df.loc[:, ['days_active_lifetime_std', 'transaction_count_iap_lifetime_std', 'match_win_rate']] = df.loc[:, ['days_active_lifetime_std', 'transaction_count_iap_lifetime_std', 'match_win_rate']].fillna(0)
    # drop columns
    df.drop(columns=drop_cols, inplace=True)

    return df


def main():
    source_dir = 'cleaned_data'
    dest_dir = 'aggregated_data'
    registration_train = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), source_dir, 'registration_data_training_labeled.csv'), index_col=0)
    prev_lives_train = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), source_dir, 'previous_lives_training_data_labeled.csv'), index_col=0)
    ########## Training data ##########
    # transform date for registration data
    registration_train = transform_date(registration_train, 'registration_time_utc', drop_original=False)
    registration_train.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), dest_dir, 'registration_data_training_labeled_transformed.csv'))
    # transform date for previous lives data
    prev_lives_train = transform_date(prev_lives_train, 'registration_date', drop_original=False)
    prev_lives_train.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), dest_dir, 'previous_lives_training_data_labeled_transformed.csv'))
    # aggregate previous lives data
    agg_df = create_agg_features(prev_lives_train)
    agg_df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), dest_dir, 'aggregated_previous_lives_training_data.csv'))

    # merge registration and aggregated previous lives data
    merged_df = pd.merge(registration_train, agg_df, left_index=True, right_index=True)
    merged_df = create_features_on_merged_data(merged_df, drop_cols=['registration_time_utc', 'first_registration_date_prev_lives', 'last_registration_date_prev_lives'])
    merged_df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), dest_dir, 'merged_data_training.csv'))
    
    ########## Test data ##########
    registration_test = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), source_dir, 'registration_data_test_labeled.csv'), index_col=0)
    prev_lives_test = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), source_dir, 'previous_lives_test_data_labeled.csv'), index_col=0)

    # transform date for registration data
    registration_test = transform_date(registration_test, 'registration_time_utc', drop_original=False)
    registration_test.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), dest_dir, 'registration_data_test_labeled_transformed.csv'))
    # transform date for previous lives data
    prev_lives_test = transform_date(prev_lives_test, 'registration_date', drop_original=False)
    prev_lives_test.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), dest_dir, 'previous_lives_test_data_labeled_transformed.csv'))
    # aggregate previous lives data
    agg_df = create_agg_features(prev_lives_test)
    agg_df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), dest_dir, 'aggregated_previous_lives_test_data.csv'))
    # merge registration and aggregated previous lives data
    merged_df = pd.merge(registration_test, agg_df, left_index=True, right_index=True)
    merged_df = create_features_on_merged_data(merged_df, drop_cols=['registration_time_utc', 'first_registration_date_prev_lives', 'last_registration_date_prev_lives'])
    merged_df.to_csv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), dest_dir, 'merged_data_test.csv'))    

if __name__ == "__main__":
    main()