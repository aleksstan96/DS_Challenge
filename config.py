nominal_features_registration = ['registration_platform_specific', 'registration_country', 'registration_store', 'registration_channel_detailed', 'registration_device_type', 'registration_device_manufacturer']
nominal_features_prev_lives = ['registration_platform_specific', 'registration_country', 'registration_store', 'registration_channel_detailed', 'is_payer_lifetime', 'is_rewarded_video_watcher_lifetime']

country_threshold = 0.0018 #0.00383
manufacturer_threshold = 0.01

def mode_value(x):
    return x.mode().iloc[0] if not x.mode().empty else None

aggs = {}
aggs['year'] = ['nunique', mode_value]
aggs['month'] = ['nunique', mode_value]
aggs['week'] = ['nunique', mode_value]
aggs['dayofweek'] = ['nunique', mode_value]
aggs['dayofmonth'] = ['nunique', mode_value]
aggs['weekend'] = ['nunique', mode_value]

aggs['registration_date'] = ['min', 'max']

aggs['registration_country'] = ['nunique']
aggs['registration_channel_detailed'] = ['nunique']
aggs['registration_store'] = ['nunique']
aggs['registration_platform_specific'] = ['nunique']

aggs['is_payer_lifetime'] = ['sum']
aggs['is_rewarded_video_watcher_lifetime'] = ['sum']
aggs['days_active_lifetime'] = ['sum', 'mean', 'max', 'min', 'std']
aggs['transaction_count_iap_lifetime'] = ['sum', 'mean', 'max', 'min', 'std']
